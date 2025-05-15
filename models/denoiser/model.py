import math
import torch
from torch import nn

from utils import SimpleGate
from .conditional_naf import ConditionalNAFBlock


# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()

        latent_channel = 4
        width = 32

        self.latent_channel = latent_channel
        self.upscale = 1
        fourier_dim = width
        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)
        time_dim = width * 4

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim * 2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim),
        )

        self.intro = nn.Conv2d(
            in_channels=latent_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=latent_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in [2, 2, 4, 8]:
            self.encoders.append(
                nn.Sequential(
                    *[ConditionalNAFBlock(chan, time_dim) for _ in range(num)]
                )
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[ConditionalNAFBlock(chan, time_dim) for _ in range(8)]
        )

        for num in [2, 2, 2, 2]:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[ConditionalNAFBlock(chan, time_dim) for _ in range(num)]
                )
            )

    def forward(self, latents, timesteps, cond=None):
        if isinstance(timesteps, int) or isinstance(timesteps, float):
            timesteps = torch.tensor([timesteps])

        x = latents
        # x = torch.cat([x, cond], dim=1)
        B, C, H, W = x.shape

        t = self.time_mlp(timesteps)
        x = self.intro(x)

        enc_skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x, _ = encoder([x, t])
            enc_skips.append(x)
            x = down(x)

        x, _ = self.middle_blks([x, t])
        for decoder, up, enc_skip in zip(self.decoders, self.ups, enc_skips[::-1]):
            x = up(x)
            x = x + enc_skip
            x, _ = decoder([x, t])

        x = self.ending(x)
        x = x[..., :H, :W]

        return x
