import os
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.utils import save_image
from diffusers import AutoencoderKL, DDIMScheduler, DDIMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm.auto import tqdm
import argparse

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
parser.add_argument("--num_epoch", type=int, default=100, help="A number of epoch")
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
    "--save_model_epoch",
    type=int,
    default=10,
    help="A number of epoch to save current model",
)
parser.add_argument(
    "--save_image_epoch",
    type=int,
    default=1,
    help="A number of epoch to save sample images",
)
args = parser.parse_args()


num_epoch = args.num_epoch

os.makedirs("./checkpoints/denoiser/%s" % args.name, exist_ok=True)
os.makedirs("./output/denoiser/%s" % args.name, exist_ok=True)


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


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

    # 초기 latent: 표준 정규분포에서 샘플링
    latents = torch.randn(
        (args.sample_size, latent_channels, latent_res, latent_res)
    )

    # DDIM 스케줄러 설정
    scheduler.set_timesteps(num_inference_steps)

    for t in scheduler.timesteps:
        # 모델의 노이즈 예측
        t_batch = torch.full((args.sample_size,), t)
        noise_pred = model(latents, t_batch)

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


def train_loop(model, noise_scheduler, vae, optimizer, train_dataloader, lr_scheduler, accelerator):
    global_step = 0

    for epoch in range(num_epoch):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch
            clean_latents = (
                vae.encode(clean_images).latent_dist.sample() * 0.18215
            )

            noise = torch.randn(clean_latents.shape).to(accelerator.device)
            bs = clean_latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                dtype=torch.int64,
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

        # 각 에포크가 끝난 후 evaluate()와 몇 가지 데모 이미지를 선택적으로 샘플링하고 모델을 저장합니다.
        if (epoch + 1) % args.save_image_epoch == 0 or epoch + 1 == num_epoch:
            ddim_sample(model, vae, noise_scheduler, epoch + 1)

        if (epoch + 1) % args.save_model_epoch == 0 or epoch + 1 == num_epoch:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                "./checkpoints/denoiser/%s/%d.pt" % (args.name, epoch + 1),
            )


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
vae = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-2-1", subfolder="vae"
)
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000, beta_schedule="scaled_linear", prediction_type="epsilon"
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=(len(train_dataloader) * num_epoch),
)

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
train_dataloader, model, optimizer, lr_scheduler = accelerator.prepare(
    train_dataloader, model, optimizer, lr_scheduler
)
vae = vae.to(accelerator.device)

train_loop(model, noise_scheduler, vae, optimizer, train_dataloader, lr_scheduler, accelerator)
