import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
import argparse
import sys
import gc
import pyiqa

from dataset import KfaceDataset
from models.refiner import FacialRefiner
from models.cr.model import CoarseRestoration

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name",
    type=str,
    default="0",
    help="A number for checkpoints and output path names",
)
parser.add_argument("--num_epoch", type=int, default=24, help="A number of epoch")
parser.add_argument(
    "--batch_size", type=int, default=8, help="A batch size of training dataset"
)
parser.add_argument(
    "--sample_size", type=int, default=8, help="A batch size of training dataset"
)
parser.add_argument(
    "--image_res",
    type=int,
    default=128,
    help="Width and height of images used for training",
)
parser.add_argument(
    "--cr_ckpt",
    type=str,
    required=False,
    default="checkpoints/cr/24.pt",
    help="A path of checkpoint (.pt) of the CR module",
)
parser.add_argument(
    "--idc_ckpt",
    type=str,
    required=False,
    default="checkpoints/idc/24.pt",
    help="A path of checkpoint (.pt) of the CR module",
)
parser.add_argument(
    "--denoiser_ckpt",
    type=str,
    required=False,
    default="checkpoints/denoiser/295.pt",
    help="A path of checkpoint (.pt) of the denoiser",
)
parser.add_argument(
    "--save_model_epoch",
    type=int,
    default=4,
    help="A number of epoch to save current model",
)
parser.add_argument(
    "--save_image_epoch",
    type=int,
    default=1,
    help="A number of epoch to save sample images",
)
args = parser.parse_args()


os.makedirs("./checkpoints/refiner/%s" % args.name, exist_ok=True)
os.makedirs("./output/refiner/%s/val" % args.name, exist_ok=True)
torch.manual_seed(0)


@torch.no_grad()
def ddim_sample(
    ln_face,
    unet,
    vae,
    cr_module,
    scheduler,
    num_inference_steps=50,
):
    latent_res = args.image_res // 8
    latent_channels = 4

    latents = torch.randn(
        (args.batch_size, latent_channels, latent_res, latent_res)
    ).to(accelerator.device)

    cr_face = cr_module(ln_face)
    cr_latent = (
        vae.encode(
            F.interpolate(cr_face, args.image_res, mode="bicubic")
        ).latent_dist.sample()
        * 0.18215
    )
    scheduler.set_timesteps(num_inference_steps)

    for t in scheduler.timesteps:
        t_batch = torch.full((args.batch_size,), t).to(accelerator.device)
        # print("🍊", latents.shape, t_batch.shape, cr_face.shape, cr_latent.shape)
        noise_pred = unet(latents, t_batch, cr_face, cr_latent)

        latents = scheduler.step(noise_pred, t, latents, eta=0.0).prev_sample

    images = vae.decode(latents / 0.18215).sample

    return images


def train_loop(
    model,
    vae,
    cr_module,
    noise_scheduler,
    optimizer,
    train_dataloader,
    lr_scheduler,
    epoch,
    accelerator,
):
    progress_bar = tqdm(
        total=len(train_dataloader), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description(f"Epoch {epoch}")

    global_step = 0
    model.train()

    for ln_face, hf_face, _ in train_dataloader:
        train_loss = 0.0
        hf_latent = (
            vae.encode(
                F.interpolate(hf_face, args.image_res, mode="bicubic")
            ).latent_dist.sample()
            * 0.18215
        )
        noise = torch.randn(hf_latent.shape).to(accelerator.device)
        bs = hf_latent.shape[0]
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bs,)
        ).to(accelerator.device)

        noisy_latent = noise_scheduler.add_noise(hf_latent, noise, timesteps)

        cr_face = cr_module(ln_face)
        cr_latent = (
            vae.encode(
                F.interpolate(cr_face, args.image_res, mode="bicubic")
            ).latent_dist.sample()
            * 0.18215
        )

        with accelerator.accumulate(model):
            noise_pred = model(noisy_latent, timesteps, cr_face, cr_latent)
            loss = F.mse_loss(noise_pred, noise)
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
        train_loss += avg_loss.item()

        progress_bar.update(1)
        global_step += 1
        accelerator.log({"train_loss": train_loss}, step=global_step)
        logs = {
            "loss": loss.detach().item(),
            "lr": lr_scheduler.get_last_lr()[0],
            "step": global_step,
        }
        progress_bar.set_postfix(**logs)

    progress_bar.close()
    accelerator.wait_for_everyone()

    if epoch % args.save_model_epoch == 0 or epoch == args.num_epoch - 1:
        accelerator.save_state("./checkpoints/refiner/%s/%d" % (args.name, epoch))

    gc.collect()
    torch.cuda.empty_cache()


def val_loop(
    model, vae, cr_module, noise_scheduler, val_dataloader, epoch, accelerator, metrics
):
    progress_bar = tqdm(
        total=len(val_dataloader), disable=not accelerator.is_local_main_process
    )
    global_step = 0
    scores = [0.0, 0.0, 0.0, 0.0]
    model.eval()

    for idx, (ln_face, hf_face, _) in enumerate(val_dataloader):
        unet = accelerator.unwrap_model(model)
        result = ddim_sample(ln_face, unet, vae, cr_module, noise_scheduler)
        # result = F.interpolate(result, 128, mode="bicubic")

        result_normalized = (result - result.min()) / (result.max() - result.min())
        sample_hf_normalized = (hf_face - hf_face.min()) / (
            hf_face.max() - hf_face.min()
        )

        for i in range(len(metrics)):
            score = metrics[i](result_normalized, sample_hf_normalized).mean().item()
            scores[i] += score

        progress_bar.update(1)
        logs = {
            "PSNR": scores[0] / (idx + 1),
            "SSIM": scores[1] / (idx + 1),
            "LPIPS": scores[2] / (idx + 1),
        }
        progress_bar.set_postfix(**logs)
        global_step += 1

        if accelerator.is_local_main_process and idx == 0:
            print("🍊 saving the sample images")
            save_image(
                torch.concat(
                    [
                        ln_face[: args.sample_size],
                        result[: args.sample_size],
                        hf_face[: args.sample_size],
                    ]
                ),
                os.path.join("output/refiner/%s/val/%d.png" % (args.name, epoch)),
                nrow=args.sample_size,
                normalize=True,
                value_range=(0, 1),
            )

    if accelerator.is_local_main_process:
        print(
            "✅ PSNR=%.4f / SSIM=%.4f / LPIPS=%.4f / NIQE=%.4f"
            % (
                scores[0] / len(val_dataloader),
                scores[1] / len(val_dataloader),
                scores[2] / len(val_dataloader),
                scores[3] / len(val_dataloader),
            ),
            file=sys.stderr,
        )


train_dataset = KfaceDataset(
    dataroot="../../datasets/kface",
    use="train",
)
val_dataset = KfaceDataset(
    dataroot="../../datasets/kface",
    use="val",
)
train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=args.batch_size, shuffle=True
)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size)

model = FacialRefiner(args.image_res // 8, args.idc_ckpt, args.denoiser_ckpt)

ddpm_scheduler = DDPMScheduler(num_train_timesteps=1000)
ddim_scheduler = DDIMScheduler()
ddim_scheduler.set_timesteps(50)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=(len(train_dataloader) * args.num_epoch),
)

accelerator = Accelerator()
train_dataloader, model, optimizer, lr_scheduler = accelerator.prepare(
    train_dataloader, model, optimizer, lr_scheduler
)
val_dataloader, model = accelerator.prepare(val_dataloader, model)

vae = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-2-1", subfolder="vae"
).to(accelerator.device)
cr_module = CoarseRestoration().to(accelerator.device)
cr_module.load_state_dict(torch.load(args.cr_ckpt)["model_state_dict"])
cr_module.eval()

metrics = []
for i, metric_name in enumerate(["psnr", "ssim", "lpips", "niqe"]):
    metric = pyiqa.create_metric(metric_name, device=accelerator.device)
    metrics.append(metric)

for epoch in range(args.num_epoch):
    train_loop(
        model,
        vae,
        cr_module,
        ddpm_scheduler,
        optimizer,
        train_dataloader,
        lr_scheduler,
        epoch,
        accelerator,
    )
    val_loop(
        model,
        vae,
        cr_module,
        ddim_scheduler,
        val_dataloader,
        epoch,
        accelerator,
        metrics,
    )
