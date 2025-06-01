import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
import argparse
import sys
import gc

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
    default="checkpoints/denoiser/03_w128/295.pt",
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
    cr_latent = vae.encode(cr_face).latent_dist.sample() * 0.18215

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
        hf_latent = vae.encode(hf_face).latent_dist.sample() * 0.18215

        noise = torch.randn(hf_latent.shape).to(accelerator.device)
        bs = hf_latent.shape[0]
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bs,)
        ).to(accelerator.device)

        noisy_latent = noise_scheduler.add_noise(hf_latent, noise, timesteps)

        cr_face = cr_module(ln_face)
        cr_latent = vae.encode(cr_face).latent_dist.sample() * 0.18215

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

    if epoch % args.save_model_epoch == 0 or epoch == args.num_epoch - 1:
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


train_dataset = KfaceDataset(
    dataroot="../../datasets/kface",
    use="train",
)
train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=args.batch_size, shuffle=True
)

model = FacialRefiner(args.image_res // 8, args.idc_ckpt, args.denoiser_ckpt)

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
