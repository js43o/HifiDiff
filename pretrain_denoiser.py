import os
import torch
from torch.nn import functional as F
from torchvision.utils import save_image
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
from collections import OrderedDict
import argparse
import sys
import gc

from models.denoiser.model import Denoiser
from dataset import KfaceHRDataset, CelebAHQDataset


parser = argparse.ArgumentParser()
parser.add_argument(
    "--name",
    type=str,
    default="0",
    help="A number for checkpoints and output path names",
)
parser.add_argument("--gpu", type=str, default="0", help="Which GPU to use")
parser.add_argument("--num_epoch", type=int, default=300, help="A number of epoch")
parser.add_argument(
    "--batch_size", type=int, default=8, help="A batch size of training dataset"
)
parser.add_argument(
    "--sample_size", type=int, default=8, help="The number of sampling images"
)
parser.add_argument(
    "--image_res",
    type=int,
    default=128,
    help="Width and height of images used for pre-training",
)
parser.add_argument(
    "--ckpt",
    type=str,
    required=False,
    help="A path of checkpoint (.pt) to continue training",
)
parser.add_argument(
    "--save_model_epoch",
    type=int,
    default=5,
    help="A number of epoch to save current model",
)
parser.add_argument(
    "--save_image_epoch",
    type=int,
    default=1,
    help="A number of epoch to save sample images",
)
args = parser.parse_args()


os.makedirs("./checkpoints/denoiser/%s" % args.name, exist_ok=True)
os.makedirs("./output/denoiser/%s" % args.name, exist_ok=True)
torch.manual_seed(0)


@torch.no_grad()
def ddim_sample(
    model,
    vae,
    scheduler,
    epoch,
    num_inference_steps=50,
):
    latent_res = args.image_res // 8
    latent_channels = 4

    unet = accelerator.unwrap_model(model)

    # 초기 latent: 표준 정규분포에서 샘플링
    latents = torch.randn(
        (args.sample_size, latent_channels, latent_res, latent_res)
    ).to(accelerator.device)

    # DDIM 스케줄러 설정
    scheduler.set_timesteps(num_inference_steps)

    for t in scheduler.timesteps:
        # 모델의 노이즈 예측
        t_batch = torch.full((args.sample_size,), t).to(accelerator.device)
        noise_pred = unet(latents, t_batch)

        # DDIM step
        latents = scheduler.step(noise_pred, t, latents, eta=0.0).prev_sample

    # latent → 이미지 복원
    images = vae.decode(latents / 0.18215).sample  # LDM scale factor 보정
    save_image(
        images,
        os.path.join("output/denoiser/%s" % args.name, "%d.png" % epoch),
        nrow=2,
        normalize=True,
        value_range=(0, 1),
    )


def train_loop(
    model,
    noise_scheduler,
    vae,
    optimizer,
    train_dataloader,
    lr_scheduler,
    accelerator,
    start_epoch=0,
):
    global_step = 0

    for epoch in range(start_epoch, args.num_epoch):
        model.train()
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            clean_images = batch
            clean_latents = vae.encode(clean_images).latent_dist.sample() * 0.18215

            noise = torch.randn(clean_latents.shape).to(accelerator.device)
            bs = clean_latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,)
            ).to(accelerator.device)

            # 각 타임스텝의 노이즈 크기에 따라 깨끗한 이미지에 노이즈를 추가합니다. (forward diffusion)
            noisy_latents = noise_scheduler.add_noise(clean_latents, noise, timesteps)

            with accelerator.accumulate(model):
                # 노이즈를 반복적으로 예측합니다.
                noise_pred = model(noisy_latents, timesteps)
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

                avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
                train_loss += avg_loss.item()

            accelerator.log({"train_loss": train_loss}, step=global_step)

        progress_bar.close()
        accelerator.wait_for_everyone()

        if epoch % args.save_model_epoch == 0 or epoch == args.num_epoch - 1:
            accelerator.save_state("./checkpoints/denoiser/%s/%d" % (args.name, epoch))

        if epoch % args.save_image_epoch == 0 or epoch == args.num_epoch - 1:
            ddim_sample(model, vae, noise_scheduler, epoch)

        gc.collect()
        torch.cuda.empty_cache()

    accelerator.end_training()


train_dataset_kface = KfaceHRDataset(
    dataroot="../../datasets/kface", res=args.image_res
)
train_dataset_celeba = CelebAHQDataset(
    dataroot="../../datasets/celeba_hq_256", res=args.image_res
)
train_dataset = torch.utils.data.ConcatDataset(
    [train_dataset_kface, train_dataset_celeba]
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True
)

model = Denoiser(latent_res=args.image_res // 8)
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae")
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000, beta_schedule="scaled_linear", prediction_type="epsilon"
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=(len(train_dataloader) * args.num_epoch),
)
start_epoch = 0

"""
if args.ckpt is not None:
    checkpoint = torch.load(args.ckpt)
    model_state_dict = checkpoint["model_state_dict"]
    optimizer_state_dict = checkpoint["optimizer_state_dict"]

    if list(model_state_dict.keys())[0].startswith("module"):
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        model_state_dict = new_state_dict

    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    start_epoch = int(checkpoint["epoch"])
"""

accelerator = Accelerator()
train_dataloader, model, optimizer, lr_scheduler = accelerator.prepare(
    train_dataloader, model, optimizer, lr_scheduler
)
vae = vae.to(accelerator.device)

train_loop(
    model,
    noise_scheduler,
    vae,
    optimizer,
    train_dataloader,
    lr_scheduler,
    accelerator,
    start_epoch,
)
