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
import pyiqa
from safetensors.torch import load_file

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
    "--refiner_ckpt",
    type=str,
    default="checkpoints/refiner/0/24.pt",
    help="A path of checkpoint (.pt) of the refiner (denoiser and FPG module)",
)
parser.add_argument(
    "--save_image_iters",
    type=int,
    default=10,
    help="Width and height of images used for training",
)
args = parser.parse_args()

os.makedirs("output/refiner/%s/test" % args.name, exist_ok=True)


@torch.no_grad()
def ddim_sample(
    ln_face,
    model,
    vae,
    cr_module,
    scheduler,
    num_inference_steps=50,
):
    latent_res = args.image_res // 8
    latent_channels = 4
    bs = ln_face.shape[0]

    model.eval()

    latent = torch.randn((bs, latent_channels, latent_res, latent_res)).to(
        accelerator.device
    )

    cr_face = cr_module(ln_face)
    cr_latent = (
        vae.encode(
            F.interpolate(cr_face, args.image_res, mode="bicubic")
        ).latent_dist.sample()
        * 0.18215
    )

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
    scores = [0.0, 0.0, 0.0, 0.0]
    metrics = []
    model.eval()

    for i, metric_name in enumerate(["psnr", "ssim", "lpips", "niqe"]):
        metric = pyiqa.create_metric(metric_name, device=accelerator.device)
        metrics.append(metric)

    for idx, (ln_face, hf_face, _) in enumerate(val_dataloader):
        result = ddim_sample(ln_face, model, vae, cr_module, noise_scheduler)
        result = F.interpolate(result, 128, mode="bicubic")

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

        if idx % args.save_image_iters == 0:
            save_image(
                torch.concat([ln_face, result, hf_face]),
                os.path.join("output/refiner/%s/test/%d.png" % (args.name, idx)),
                nrow=4,
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


val_dataset = KfaceDataset(
    dataroot="../../datasets/kface",
    use="val",
)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size)

model = FacialRefiner(latent_res=args.image_res // 8)
refiner_weights = load_file(args.refiner_ckpt)
model.load_state_dict(refiner_weights)

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_schedule="scaled_linear",
    prediction_type="epsilon",
    clip_sample_range=2.0,
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
