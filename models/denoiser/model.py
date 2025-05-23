import math
import torch
from torch import nn

from utils import SimpleGate
from .conditional_naf import ConditionalNAFBlock
from ..fpg.hca import HybridCrossAttention


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
    def __init__(self, latent_res):
        super().__init__()

        self.latent_channel = 4
        self.latent_res = latent_res
        self.width = 32

        fourier_dim = self.width
        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)
        time_dim = self.width * 4

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim * 2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim),
        )

        self.intro = nn.Conv2d(
            in_channels=self.latent_channel,
            out_channels=self.width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.ending = nn.Conv2d(
            in_channels=self.width,
            out_channels=self.latent_channel,
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
        self.hcas = nn.ModuleList()

        chan = self.width
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
        self.idc_conv = nn.Conv2d(
            2048, (self.width * (2**4)) * (self.latent_res // (2**4)) ** 2, (1, 1)
        )
        self.hcas.append(HybridCrossAttention(chan, self.latent_res // (2**4)))

        for idx, num in enumerate([2, 2, 2, 2]):
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
            self.hcas.append(
                HybridCrossAttention(chan, self.latent_res // (2 ** (4 - idx)))
            )

    def forward(self, latents, timesteps, facial_priors=None, identity_embedding=None):
        if isinstance(timesteps, int) or isinstance(timesteps, float):
            timesteps = torch.tensor([timesteps])

        x = latents
        batch, _, height, width = x.shape

        t = self.time_mlp(timesteps)
        x = self.intro(x)

        enc_skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x, _ = encoder([x, t])
            enc_skips.append(x)
            x = down(x)

        x, _ = self.middle_blks([x, t])

        if (identity_embedding and facial_prior) is not None:
            # main training
            idc = self.idc_conv(identity_embedding)
            x = x + idc.reshape(batch, *x.shape[1:])
            x = self.hcas[0](facial_priors[0], x)

            for decoder, up, hca, facial_prior, enc_skip in zip(
                self.decoders,
                self.ups,
                self.hcas[1:],
                facial_priors[1:],
                enc_skips[::-1],
            ):
                x = up(x)
                x = x + enc_skip
                x, _ = decoder([x, t])
                x = hca(facial_prior, x)
        else:
            # pre-training
            for decoder, up in zip(self.decoders, self.ups):
                x = up(x)
                x, _ = decoder([x, t])

        x = self.ending(x)
        x = x[..., :height, :width]

        return x
