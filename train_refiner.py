import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from diffusers import AutoencoderKL, DDIMScheduler
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
    "--batch_size", type=int, default=4, help="A batch size of training dataset"
)
parser.add_argument(
    "--image_res",
    type=int,
    default=512,
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
    default="checkpoints/denoiser/40.pt",
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
os.makedirs("./output/refiner/%s" % args.name, exist_ok=True)
torch.manual_seed(0)


@torch.no_grad()
def ddim_sample(
    ln_face,
    model,
    vae,
    scheduler,
    num_inference_steps=50,
):
    latent_res = args.image_res // 8
    latent_channels = 4
    bs = ln_face.shape[0]

    model.eval()

    # 초기 latent: 표준 정규분포에서 샘플링
    latent = torch.randn((bs, latent_channels, latent_res, latent_res)).to(
        accelerator.device
    )

    cr_face = cr_module(ln_face)
    cr_face_upscaled = F.interpolate(cr_face, 512, mode="bicubic")
    cr_latent = vae.encode(cr_face_upscaled).latent_dist.sample() * 0.18215

    scheduler.set_timesteps(num_inference_steps)

    for t in scheduler.timesteps:
        t_batch = torch.full((bs,), t).to(accelerator.device)
        noise_pred = model(latent, t_batch, cr_face, cr_latent)

        latent = scheduler.step(noise_pred, t, latent, eta=0.0).prev_sample

    images = vae.decode(latent / 0.18215).sample

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
    acc_loss = 0
    model.train()

    for ln_face, hf_face, _ in train_dataloader:
        hf_face_upscaled = F.interpolate(hf_face, 512, mode="bicubic")
        hf_latent = vae.encode(hf_face_upscaled).latent_dist.sample() * 0.18215

        noise = torch.randn(hf_latent.shape).to(accelerator.device)
        bs = hf_latent.shape[0]
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bs,)
        ).to(accelerator.device)

        noisy_latent = noise_scheduler.add_noise(hf_latent, noise, timesteps)

        cr_face = cr_module(ln_face)
        cr_face_upscaled = F.interpolate(cr_face, 512, mode="bicubic")
        cr_latent = vae.encode(cr_face_upscaled).latent_dist.sample() * 0.18215

        noise_pred = model(noisy_latent, timesteps, cr_face, cr_latent)
        loss = F.mse_loss(noise_pred, noise)
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress_bar.update(1)
        logs = {
            "loss": loss.detach().item(),
            "lr": lr_scheduler.get_last_lr()[0],
            "step": global_step,
        }
        progress_bar.set_postfix(**logs)
        global_step += 1

        if accelerator.is_local_main_process:
            acc_loss += loss.detach().item()

    if accelerator.is_local_main_process:
        print(
            "🔄 Training loss = %.4f" % (acc_loss / len(train_dataloader)),
            file=sys.stderr,
        )

    if epoch % args.save_model_epoch == 0 or epoch == args.num_epoch:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            "./checkpoints/refiner/%s/%d.pt" % (args.name, epoch),
        )

    gc.collect()
    torch.cuda.empty_cache()


def val_loop(
    model, vae, cr_module, noise_scheduler, val_dataloader, epoch, accelerator
):
    progress_bar = tqdm(
        total=len(val_dataloader), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description(f"Epoch {epoch}")
    global_step = 0

    acc_loss = 0
    sample_faces = (None, None)
    model.eval()

    for idx, (ln_face, hf_face, _) in enumerate(val_dataloader):
        hf_face_upscaled = F.interpolate(hf_face, 512, mode="bicubic")
        hf_latent = vae.encode(hf_face_upscaled).latent_dist.sample() * 0.18215

        noise = torch.randn(hf_latent.shape).to(accelerator.device)
        bs = hf_latent.shape[0]
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bs,)
        ).to(accelerator.device)

        noisy_latent = noise_scheduler.add_noise(hf_latent, noise, timesteps)

        cr_face = cr_module(ln_face)
        cr_face_upscaled = F.interpolate(cr_face, 512, mode="bicubic")
        cr_latent = vae.encode(cr_face_upscaled).latent_dist.sample() * 0.18215

        noise_pred = model(noisy_latent, timesteps, cr_face, cr_latent)
        loss = F.mse_loss(noise_pred, noise)

        progress_bar.update(1)
        logs = {
            "loss": loss.detach().item(),
            "step": global_step,
        }
        progress_bar.set_postfix(**logs)
        global_step += 1

        if accelerator.is_local_main_process:
            acc_loss += loss.detach().item()
            if idx == 0:
                sample_faces = (ln_face, hf_face)

    if accelerator.is_local_main_process:
        print(
            "✅ Validation loss = %.4f" % (acc_loss / len(val_dataloader)),
            file=sys.stderr,
        )

        sample_ln, sample_hf = sample_faces
        result = ddim_sample(sample_ln, model, vae, noise_scheduler)
        result = F.interpolate(result, 128, mode="bicubic")

        result_normalized = (result - result.min()) / (result.max() - result.min())
        sample_hf_normalized = (sample_hf - sample_hf.min()) / (
            sample_hf.max() - sample_hf.min()
        )

        for metric_name in ["psnr", "ssim", "lpips", "niqe"]:
            metric = pyiqa.create_metric(metric_name, device=accelerator.device)
            score = metric(result_normalized, sample_hf_normalized).mean().item()
            print("🍊 %s = %.4f" % (metric_name, score))

        save_image(
            torch.concat([sample_ln, result, sample_hf]),
            os.path.join("output/refiner/%s" % args.name, "%d.png" % epoch),
            nrow=2,
            normalize=True,
            value_range=(0, 1),
        )

    gc.collect()
    torch.cuda.empty_cache()


train_dataset = KfaceDataset(
    dataroot="../../datasets/kface",
    use="train",
)
# val_dataset = KfaceDataset(
#     dataroot="../../datasets/kface",
#     use="val",
# )

train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=args.batch_size, shuffle=True
)
# val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size)

model = FacialRefiner(args.idc_ckpt, args.denoiser_ckpt)

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000, beta_schedule="scaled_linear", prediction_type="epsilon"
)
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
# val_dataloader = accelerator.prepare(val_dataloader)

vae = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-2-1", subfolder="vae"
).to(accelerator.device)
cr_module = CoarseRestoration().to(accelerator.device)
cr_module.load_state_dict(torch.load(args.cr_ckpt)["model_state_dict"])
cr_module.eval()

for epoch in range(args.num_epoch):
    train_loop(
        model,
        vae,
        cr_module,
        noise_scheduler,
        optimizer,
        train_dataloader,
        lr_scheduler,
        epoch,
        accelerator,
    )
    # val_loop(model, vae, cr_module, noise_scheduler, val_dataloader, epoch, accelerator)
