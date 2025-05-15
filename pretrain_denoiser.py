import os
import torch
from PIL import Image
from torch.nn import functional as F
from diffusers import AutoencoderKL, DDIMScheduler, DDIMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
from torchvision.utils import save_image

from models.denoiser.model import Denoiser
from dataset import KfaceDataset_HROnly

os.environ["CUDA_VISIBLE_DEVICES"] = "5"


device = "cuda"
save_model_epochs = 10
save_image_epochs = 1
num_epochs = 50


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(epoch, pipeline):
    # 랜덤한 노이즈로 부터 이미지를 추출합니다.(이는 역전파 diffusion 과정입니다.)
    # 기본 파이프라인 출력 형태는 `List[PIL.Image]` 입니다.
    eval_batch_size = 8
    seed = 0
    output_dir = "output/denoiser"

    images = pipeline(
        batch_size=eval_batch_size,
        generator=torch.manual_seed(seed),
    ).images

    # 이미지들을 그리드로 만들어줍니다.
    image_grid = make_grid(images, rows=4, cols=4)

    # 이미지들을 저장합니다.
    test_dir = os.path.join(output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


@torch.no_grad()
def ddim_sample(
    model,  # 훈련된 커스텀 모델
    vae,  # VAE 디코더
    epoch,
    num_images=8,  # 배치 크기
    num_inference_steps=150,
    save_dir="output/denoiser",
):
    # latent 크기
    latent_height = 16
    latent_width = 16
    latent_channels = 4  # 일반적으로 LDM latent는 4채널

    # 초기 latent: 표준 정규분포에서 샘플링
    latents = torch.randn(
        (num_images, latent_channels, latent_height, latent_width), device=device
    )

    # DDIM 스케줄러 설정
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(num_inference_steps, device=device)

    for t in scheduler.timesteps:
        # 모델의 노이즈 예측
        t_batch = torch.full((num_images,), t, device=device)
        noise_pred = model(latents, t_batch)

        # DDIM step
        latents = scheduler.step(noise_pred, t, latents, eta=0.0).prev_sample

    # latent → 이미지 복원
    images = vae.decode(latents / 0.18215).sample  # LDM scale factor 보정
    save_image(
        images,
        os.path.join(save_dir, "%d.png" % epoch),
        nrow=2,
        normalize=True,
        value_range=(-1, 1),
    )


def train_loop(model, noise_scheduler, vae, optimizer, train_dataloader, lr_scheduler):
    global_step = 0

    for epoch in range(num_epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch.to(device)
            clean_latents = (
                vae.encode(clean_images).latent_dist.sample() * 0.18215
            ).to(
                device
            )  # [B, 4, 16, 16]
            # 이미지에 더할 노이즈를 샘플링합니다.
            noise = torch.randn(clean_latents.shape, device=clean_latents.device)
            bs = clean_latents.shape[0]

            # 각 이미지를 위한 랜덤한 타임스텝(timestep)을 샘플링합니다.
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=clean_latents.device,
                dtype=torch.int64,
            )

            # 각 타임스텝의 노이즈 크기에 따라 깨끗한 이미지에 노이즈를 추가합니다. (forward diffusion)
            noisy_latents = noise_scheduler.add_noise(clean_latents, noise, timesteps)

            # 노이즈를 반복적으로 예측합니다.
            noise_pred = model(noisy_latents, timesteps)
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()

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
        if (epoch + 1) % save_image_epochs == 0 or epoch + 1 == num_epochs:
            ddim_sample(model, vae, epoch + 1)

        if (epoch + 1) % save_model_epochs == 0 or epoch + 1 == num_epochs:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                "./checkpoints/denoiser/%d.pt" % epoch + 1,
            )


train_dataset = KfaceDataset_HROnly(
    dataroot="../../datasets/kface",
    use="train",
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=8, shuffle=True
)
model = Denoiser().to(device)
vae = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-2-1", subfolder="vae"
).to(device)
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000, beta_schedule="scaled_linear", prediction_type="epsilon"
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

train_loop(model, noise_scheduler, vae, optimizer, train_dataloader, lr_scheduler)
