import os
import gc
import argparse

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm

import wandb

from models.denoiser.model import Denoiser
from dataset_pretraining import MultiPIEHQDataset, CelebAHQDataset

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="temp")
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--sample_size", type=int, default=8)
parser.add_argument("--size", type=int, default=128)
parser.add_argument(
    "--ckpt",
    type=str,
    required=False,
    default=None,
    help="A path of accelerator checkpoint directory to continue training",
)
parser.add_argument("--save_model_epoch", type=int, default=10)
parser.add_argument("--save_image_epoch", type=int, default=1)
parser.add_argument(
    "--multipie_root",
    type=str,
    default="../../datasets/multipie_crop_patch_v2",
)
parser.add_argument(
    "--celeba_root",
    type=str,
    default="../../datasets/celeba-hq_custom-aligned",
)

args = parser.parse_args()

os.makedirs(f"./checkpoints/denoiser/{args.name}", exist_ok=True)
os.makedirs(f"./output/denoiser/{args.name}", exist_ok=True)

torch.manual_seed(0)


def prepare_vae_images(images):
    """
    Datasets return images in [0, 1].
    Stable Diffusion VAE expects images in [-1, 1].
    """
    return images.clamp(0, 1) * 2.0 - 1.0


def decode_latents(vae, latents, scaling_factor):
    images = vae.decode(latents / scaling_factor).sample
    images = ((images + 1.0) / 2.0).clamp(0, 1)
    return images


@torch.no_grad()
def encode_latents(vae, images, scaling_factor):
    images = prepare_vae_images(images)
    latents = vae.encode(images).latent_dist.sample()
    latents = latents * scaling_factor
    return latents


@torch.no_grad()
def ddim_sample(
    model,
    vae,
    scheduler,
    accelerator,
    scaling_factor,
    epoch,
    num_inference_steps=50,
):
    model.eval()

    latent_size = args.size // 8
    latent_channels = 4
    bs = args.sample_size

    unet = accelerator.unwrap_model(model)

    latents = torch.randn(
        (bs, latent_channels, latent_size, latent_size),
        device=accelerator.device,
    )

    scheduler.set_timesteps(num_inference_steps, device=accelerator.device)

    for t in scheduler.timesteps:
        t_batch = torch.full(
            (bs,),
            int(t.item()) if torch.is_tensor(t) else int(t),
            device=accelerator.device,
            dtype=torch.long,
        )

        noise_pred = unet(latents, t_batch).sample
        latents = scheduler.step(noise_pred, t, latents, eta=0.0).prev_sample

    images = decode_latents(vae, latents, scaling_factor)

    if accelerator.is_local_main_process:
        save_image(
            images,
            os.path.join(f"output/denoiser/{args.name}", f"{epoch}.png"),
            nrow=args.sample_size // int(args.sample_size**0.5),
            normalize=False,
        )


def train_loop(
    model,
    ddpm_scheduler,
    vae,
    optimizer,
    train_dataloader,
    lr_scheduler,
    ddim_scheduler,
    accelerator,
    scaling_factor,
    start_epoch=0,
):
    global_step = start_epoch * len(train_dataloader)

    for epoch in range(start_epoch, args.num_epochs):
        model.train()

        progress_bar = tqdm(
            total=len(train_dataloader),
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Epoch {epoch}")

        total_loss = 0.0

        for step, clean_images in enumerate(train_dataloader):
            with torch.no_grad():
                clean_latents = encode_latents(vae, clean_images, scaling_factor)

            noise = torch.randn_like(clean_latents)
            bs = clean_latents.shape[0]

            timesteps = torch.randint(
                0,
                ddpm_scheduler.config.num_train_timesteps,
                (bs,),
                device=accelerator.device,
                dtype=torch.long,
            )

            noisy_latents = ddpm_scheduler.add_noise(
                clean_latents,
                noise,
                timesteps,
            )

            with accelerator.accumulate(model):
                noise_pred = model(noisy_latents, timesteps).sample
                loss = F.mse_loss(noise_pred.float(), noise.float())

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            gathered_loss = accelerator.gather(loss.detach()).mean()
            total_loss += gathered_loss.item()
            global_step += 1

            accelerator.log(
                {"train_loss": gathered_loss.item()},
                step=global_step,
            )

            if accelerator.is_local_main_process:
                wandb.log(
                    {
                        "train_loss": gathered_loss.item(),
                        "avg_train_loss": total_loss / (step + 1),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "global_step": global_step,
                    }
                )

            progress_bar.update(1)
            progress_bar.set_postfix(
                loss=gathered_loss.item(),
                avg_loss=total_loss / (step + 1),
                lr=lr_scheduler.get_last_lr()[0],
                step=global_step,
            )

        progress_bar.close()
        accelerator.wait_for_everyone()

        if epoch % args.save_model_epoch == 0 or epoch == args.num_epochs - 1:
            accelerator.save_state(f"./checkpoints/denoiser/{args.name}/{epoch}")

        if epoch % args.save_image_epoch == 0 or epoch == args.num_epochs - 1:
            ddim_sample(
                model=model,
                vae=vae,
                scheduler=ddim_scheduler,
                accelerator=accelerator,
                scaling_factor=scaling_factor,
                epoch=epoch,
                num_inference_steps=50,
            )

        gc.collect()
        torch.cuda.empty_cache()

    accelerator.end_training()


def main():
    accelerator = Accelerator()

    train_dataset_multipie = MultiPIEHQDataset(
        dataroot=args.multipie_root,
        size=args.size,
    )
    train_dataset_celeba = CelebAHQDataset(
        dataroot=args.celeba_root,
        size=args.size,
    )

    train_dataset = torch.utils.data.ConcatDataset(
        [train_dataset_multipie, train_dataset_celeba]
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    model = Denoiser(latent_size=args.size // 8)

    vae = AutoencoderKL.from_pretrained(
        "Manojb/stable-diffusion-2-1-base",
        subfolder="vae",
    )

    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )

    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
        clip_sample=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_dataloader) * args.num_epochs,
    )

    train_dataloader, model, optimizer, lr_scheduler = accelerator.prepare(
        train_dataloader,
        model,
        optimizer,
        lr_scheduler,
    )

    vae = vae.to(accelerator.device)
    vae.eval()
    vae.requires_grad_(False)

    scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)

    start_epoch = 0
    if args.ckpt is not None:
        accelerator.load_state(args.ckpt)

        try:
            start_epoch = int(os.path.basename(args.ckpt.rstrip("/"))) + 1
        except ValueError:
            start_epoch = 0

    if accelerator.is_local_main_process:
        wandb.init(
            project="hifi_denoiser",
            name=args.name,
            config={
                "architecture": "HifiDiff",
                "dataset": "MultiPIEHQ + CelebAHQ",
                "epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "size": args.size,
                "scheduler": "scaled_linear_epsilon",
                "vae_scaling_factor": scaling_factor,
                "multipie_size": len(train_dataset_multipie),
                "celeba_size": len(train_dataset_celeba),
            },
        )

    train_loop(
        model=model,
        ddpm_scheduler=ddpm_scheduler,
        vae=vae,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        lr_scheduler=lr_scheduler,
        ddim_scheduler=ddim_scheduler,
        accelerator=accelerator,
        scaling_factor=scaling_factor,
        start_epoch=start_epoch,
    )

    if accelerator.is_local_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
