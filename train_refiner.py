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

import pyiqa
import wandb

from dataset_multipie import MultiPIEDataset
from models.refiner import FacialRefiner
from models.cr.model import CoarseRestoration

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="multipie")
parser.add_argument("--num_epochs", type=int, default=24)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--sample_size", type=int, default=8)
parser.add_argument("--image_res", type=int, default=128)

parser.add_argument(
    "--cr_ckpt",
    type=str,
    default="checkpoints/cr/03_crop/23.pt",
)
parser.add_argument(
    "--idc_ckpt",
    type=str,
    default="checkpoints/idc/03_crop/23.pt",
)
parser.add_argument(
    "--denoiser_ckpt",
    type=str,
    default="checkpoints/denoiser/03_crop/70/model.safetensors",
)

parser.add_argument("--save_model_epoch", type=int, default=5)
parser.add_argument("--save_image_epoch", type=int, default=1)

args = parser.parse_args()

os.makedirs(f"./checkpoints/refiner/{args.name}", exist_ok=True)
os.makedirs(f"./output/refiner/{args.name}/val", exist_ok=True)

torch.manual_seed(0)


def to_vae_range(x):
    """
    Dataset images are assumed to be in [0, 1].
    Stable Diffusion VAE expects images in [-1, 1].
    """
    return x.clamp(0, 1) * 2.0 - 1.0


def from_vae_range(x):
    """
    VAE decoded images are in approximately [-1, 1].
    Convert them back to [0, 1].
    """
    return ((x + 1.0) / 2.0).clamp(0, 1)


@torch.no_grad()
def encode_latent(vae, images, scaling_factor):
    images = F.interpolate(
        images,
        size=(args.image_res, args.image_res),
        mode="bicubic",
        align_corners=False,
    )
    images = to_vae_range(images)
    latents = vae.encode(images).latent_dist.sample()
    latents = latents * scaling_factor
    return latents


@torch.no_grad()
def ddim_sample(
    ln_face,
    unet,
    vae,
    cr_module,
    scheduler,
    accelerator,
    scaling_factor,
    num_inference_steps=50,
):
    bs = ln_face.shape[0]
    latent_res = args.image_res // 8
    latent_channels = 4

    latents = torch.randn(
        (bs, latent_channels, latent_res, latent_res),
        device=accelerator.device,
    )

    cr_face = cr_module(ln_face)
    cr_latent = encode_latent(vae, cr_face, scaling_factor)

    scheduler.set_timesteps(num_inference_steps, device=accelerator.device)

    for t in scheduler.timesteps:
        t_batch = torch.full(
            (bs,),
            int(t.item()) if torch.is_tensor(t) else int(t),
            device=accelerator.device,
            dtype=torch.long,
        )

        noise_pred = unet(latents, t_batch, cr_face, cr_latent).sample
        latents = scheduler.step(noise_pred, t, latents, eta=0.0).prev_sample

    images = vae.decode(latents / scaling_factor).sample
    images = from_vae_range(images)

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
    scaling_factor,
):
    progress_bar = tqdm(
        total=len(train_dataloader),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description(f"Epoch {epoch}")

    model.train()
    total_loss = 0.0

    for step, (ln_face, hf_face) in enumerate(train_dataloader):
        bs = hf_face.shape[0]

        with torch.no_grad():
            hf_latent = encode_latent(vae, hf_face, scaling_factor)

            cr_face = cr_module(ln_face)
            cr_latent = encode_latent(vae, cr_face, scaling_factor)

        noise = torch.randn_like(hf_latent)

        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bs,),
            device=accelerator.device,
            dtype=torch.long,
        )

        noisy_latent = noise_scheduler.add_noise(hf_latent, noise, timesteps)

        with accelerator.accumulate(model):
            noise_pred = model(noisy_latent, timesteps, cr_face, cr_latent).sample
            loss = F.mse_loss(noise_pred.float(), noise.float())

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        gathered_loss = accelerator.gather(loss.detach()).mean()
        total_loss += gathered_loss.item()

        global_step = epoch * len(train_dataloader) + step

        accelerator.log({"train_loss": gathered_loss.item()}, step=global_step)

        if accelerator.is_local_main_process:
            wandb.log(
                {
                    "train_loss": gathered_loss.item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "global_step": global_step,
                }
            )

        progress_bar.update(1)
        progress_bar.set_postfix(
            loss=gathered_loss.item(),
            avg_loss=total_loss / (step + 1),
            lr=lr_scheduler.get_last_lr()[0],
        )

    progress_bar.close()
    accelerator.wait_for_everyone()

    if epoch % args.save_model_epoch == 0 or epoch == args.num_epochs - 1:
        accelerator.save_state(f"./checkpoints/refiner/{args.name}/{epoch}")

    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def val_loop(
    model,
    vae,
    cr_module,
    noise_scheduler,
    val_dataloader,
    epoch,
    accelerator,
    metrics,
    scaling_factor,
):
    progress_bar = tqdm(
        total=len(val_dataloader),
        disable=not accelerator.is_local_main_process,
    )

    model.eval()
    scores = [0.0, 0.0, 0.0, 0.0]

    unet = accelerator.unwrap_model(model)

    for idx, (ln_face, hf_face) in enumerate(val_dataloader):
        result = ddim_sample(
            ln_face=ln_face,
            unet=unet,
            vae=vae,
            cr_module=cr_module,
            scheduler=noise_scheduler,
            accelerator=accelerator,
            scaling_factor=scaling_factor,
            num_inference_steps=50,
        )

        hf_face_128 = F.interpolate(
            hf_face,
            size=(args.image_res, args.image_res),
            mode="bicubic",
            align_corners=False,
        ).clamp(0, 1)

        for i, metric in enumerate(metrics):
            score = metric(result, hf_face_128).mean().item()
            scores[i] += score

        progress_bar.update(1)
        progress_bar.set_postfix(
            PSNR=scores[0] / (idx + 1),
            SSIM=scores[1] / (idx + 1),
            LPIPS=scores[2] / (idx + 1),
            NIQE=scores[3] / (idx + 1),
        )

        if accelerator.is_local_main_process and idx == 0:
            print("🍊 saving the sample images")

            ln_face_vis = F.interpolate(
                ln_face,
                size=(args.image_res, args.image_res),
                mode="nearest",
            ).clamp(0, 1)

            save_image(
                torch.cat(
                    [
                        ln_face_vis[: args.sample_size],
                        result[: args.sample_size],
                        hf_face_128[: args.sample_size],
                    ],
                    dim=0,
                ),
                f"output/refiner/{args.name}/val/{epoch}.png",
                nrow=args.sample_size,
                normalize=False,
            )

    progress_bar.close()

    if accelerator.is_local_main_process:
        wandb.log(
            {
                "psnr": scores[0] / len(val_dataloader),
                "ssim": scores[1] / len(val_dataloader),
                "lpips": scores[2] / len(val_dataloader),
                "niqe": scores[3] / len(val_dataloader),
                "epoch": epoch,
            }
        )


def main():
    accelerator = Accelerator()

    train_dataset = MultiPIEDataset(
        dataroot="../../datasets/multipie_crop_patch_v2",
        use="train",
        use_blind=False,
        use_patch=False,
    )
    val_dataset = MultiPIEDataset(
        dataroot="../../datasets/multipie_crop_patch_v2",
        use="test",
        use_blind=False,
        use_patch=False,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.sample_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    model = FacialRefiner(args.image_res // 8, args.idc_ckpt, args.denoiser_ckpt)

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

    train_dataloader, val_dataloader, model, optimizer, lr_scheduler = (
        accelerator.prepare(
            train_dataloader,
            val_dataloader,
            model,
            optimizer,
            lr_scheduler,
        )
    )

    vae = AutoencoderKL.from_pretrained(
        "Manojb/stable-diffusion-2-1-base",
        subfolder="vae",
    ).to(accelerator.device)
    vae.eval()
    vae.requires_grad_(False)

    scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)

    cr_module = CoarseRestoration().to(accelerator.device)
    cr_state = torch.load(args.cr_ckpt, map_location=accelerator.device)
    cr_module.load_state_dict(cr_state["model_state_dict"])
    cr_module.eval()
    cr_module.requires_grad_(False)

    if accelerator.is_local_main_process:
        wandb.init(
            project="hifi_refiner",
            name=args.name,
            config={
                "architecture": "HifiDiff",
                "dataset": "multipie_crop_patch_v2",
                "epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "image_res": args.image_res,
                "scheduler": "scaled_linear_epsilon",
                "vae_scaling_factor": scaling_factor,
            },
        )

    metrics = []
    for metric_name in ["psnr", "ssim", "lpips", "niqe"]:
        metric = pyiqa.create_metric(metric_name, device=accelerator.device)
        metrics.append(metric)

    for epoch in range(args.num_epochs):
        train_loop(
            model=model,
            vae=vae,
            cr_module=cr_module,
            noise_scheduler=ddpm_scheduler,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
            accelerator=accelerator,
            scaling_factor=scaling_factor,
        )

        if epoch % args.save_image_epoch == 0 or epoch == args.num_epochs - 1:
            val_loop(
                model=model,
                vae=vae,
                cr_module=cr_module,
                noise_scheduler=ddim_scheduler,
                val_dataloader=val_dataloader,
                epoch=epoch,
                accelerator=accelerator,
                metrics=metrics,
                scaling_factor=scaling_factor,
            )

    if accelerator.is_local_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
