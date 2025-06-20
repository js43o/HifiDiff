import os
import torch
from torch.nn import functional as F
from torchvision.utils import save_image
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
import argparse
import gc
import wandb


from models.denoiser.model import Denoiser
from dataset import KfaceCropHRDataset, KfaceHRDataset, CelebAHQDataset


parser = argparse.ArgumentParser()
parser.add_argument(
    "--name",
    type=str,
    default="0",
    help="A number for checkpoints and output path names",
)
parser.add_argument("--gpu", type=str, default="0", help="Which GPU to use")
parser.add_argument("--num_epochs", type=int, default=500, help="A number of epoch")
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
def ddim_sample(model, vae, scheduler, epoch):
    latent_res = args.image_res // 8
    latent_channels = 4

    unet = accelerator.unwrap_model(model)

    latents = torch.randn(
        (args.sample_size, latent_channels, latent_res, latent_res)
    ).to(accelerator.device)

    for t in scheduler.timesteps:
        t_batch = torch.full((args.sample_size,), t).to(accelerator.device)
        noise_pred = unet(latents, t_batch).sample

        latents = scheduler.step(noise_pred, t, latents, eta=0.0).prev_sample

    images = vae.decode(latents / 0.18215).sample
    save_image(
        images,
        os.path.join("output/denoiser/%s" % args.name, "%d.png" % epoch),
        nrow=2,
        normalize=True,
        value_range=(0, 1),
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
    start_epoch=0,
):
    global_step = 0

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            train_loss = 0.0
            clean_images = batch
            clean_latents = vae.encode(clean_images).latent_dist.sample() * 0.18215

            noise = torch.randn(clean_latents.shape).to(accelerator.device)
            bs = clean_latents.shape[0]
            timesteps = torch.randint(
                0, ddpm_scheduler.config.num_train_timesteps, (bs,)
            ).to(accelerator.device)

            # 각 타임스텝의 노이즈 크기에 따라 깨끗한 이미지에 노이즈를 추가합니다. (forward diffusion)
            noisy_latents = ddpm_scheduler.add_noise(clean_latents, noise, timesteps)

            with accelerator.accumulate(model):
                # 노이즈를 반복적으로 예측합니다.
                noise_pred = model(noisy_latents, timesteps).sample
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

            if accelerator.is_local_main_process:
                wandb.log({"train_loss": loss})

        progress_bar.close()
        accelerator.wait_for_everyone()

        if epoch % args.save_model_epoch == 0 or epoch == args.num_epochs - 1:
            accelerator.save_state(
                "./checkpoints/denoiser/%s/%d" % (args.name, epoch),
            )

        if epoch % args.save_image_epoch == 0 or epoch == args.num_epochs - 1:
            ddim_sample(model, vae, ddim_scheduler, epoch)

        gc.collect()
        torch.cuda.empty_cache()

    accelerator.end_training()


train_dataset_kface_crop = KfaceCropHRDataset(
    dataroot="../../datasets/kface_crop", res=args.image_res
)
train_dataset_celeba = CelebAHQDataset(
    dataroot="../../datasets/celeba_hq_aligned", res=args.image_res
)
train_dataset = torch.utils.data.ConcatDataset(
    [train_dataset_kface_crop, train_dataset_celeba]
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True
)

model = Denoiser(latent_res=args.image_res // 8)
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae")
ddpm_scheduler = DDPMScheduler(num_train_timesteps=1000)
ddim_scheduler = DDIMScheduler()
ddim_scheduler.set_timesteps(50)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=(len(train_dataloader) * args.num_epochs),
)
start_epoch = 0

accelerator = Accelerator()

train_dataloader, model, optimizer, lr_scheduler = accelerator.prepare(
    train_dataloader, model, optimizer, lr_scheduler
)
vae = vae.to(accelerator.device)

if args.ckpt is not None:
    accelerator.load_state(args.ckpt)
    start_epoch = int(args.ckpt.split("/")[-1])

if accelerator.is_local_main_process:
    wandb.init(
        # Set the project where this run will be logged
        project="hifi_denoiser",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"03_crop",
        # Track hyperparameters and run metadata
        config={
            "architecture": "HifiDiff",
            "dataset": "kface_crop",
            "epochs": args.num_epochs,
        },
    )

train_loop(
    model,
    ddpm_scheduler,
    vae,
    optimizer,
    train_dataloader,
    lr_scheduler,
    ddim_scheduler,
    accelerator,
    start_epoch,
)

if accelerator.is_local_main_process:
    wandb.finish()
