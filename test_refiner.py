import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from diffusers import AutoencoderKL, DDIMScheduler
from accelerate import Accelerator
from tqdm.auto import tqdm
import argparse
import sys
import gc
import pyiqa
from collections import OrderedDict

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
parser.add_argument(
    "--batch_size", type=int, default=2, help="A batch size of training dataset"
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
    "--refiner_ckpt",
    type=str,
    default="checkpoints/refiner/0/20.pt",
    help="A path of checkpoint (.pt) of the refiner (denoiser and FPG module)",
)
args = parser.parse_args()


torch.manual_seed(1)


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


def val_loop(model, vae, cr_module, noise_scheduler, val_dataloader, accelerator):
    progress_bar = tqdm(
        total=len(val_dataloader), disable=not accelerator.is_local_main_process
    )
    global_step = 0
    acc_loss = 0
    sample_faces = (None, None)
    model.eval()

    for idx, (ln_face, hf_face, _) in enumerate(val_dataloader):
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

        break

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
            os.path.join("output/refiner/%s" % args.name, "validation.png"),
            nrow=2,
            normalize=True,
            value_range=(0, 1),
        )

    gc.collect()
    torch.cuda.empty_cache()


val_dataset = KfaceDataset(
    dataroot="../../datasets/kface",
    use="val",
)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size)

model = FacialRefiner()
refiner_weights = torch.load(args.refiner_ckpt)["model_state_dict"]
temp_weights = OrderedDict()
for k, v in refiner_weights.items():
    name = k[7:]  # remove `module.`
    temp_weights[name] = v

refiner_weights = temp_weights
model.load_state_dict(refiner_weights)

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000, beta_schedule="scaled_linear", prediction_type="epsilon"
)

accelerator = Accelerator()
val_dataloader, model = accelerator.prepare(val_dataloader, model)

vae = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-2-1", subfolder="vae"
).to(accelerator.device)
cr_module = CoarseRestoration().to(accelerator.device)
cr_module.load_state_dict(torch.load(args.cr_ckpt)["model_state_dict"])
cr_module.eval()

val_loop(model, vae, cr_module, noise_scheduler, val_dataloader, accelerator)
