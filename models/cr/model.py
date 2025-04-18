import torch
from torch import nn
from models.common.naf import NAFBlock
from .stn import STNBlock


class NAF_STN_Block(nn.Module):
    def __init__(
        self, in_channel: int, in_resolution: int, num_naf: int, sampling=None
    ):
        super().__init__()

        self.nfbs = nn.Sequential(*[NAFBlock(in_channel) for _ in range(num_naf)])
        self.stn = STNBlock(in_channel, in_resolution)
        if sampling == "down":
            self.sampling = nn.Conv2d(in_channel, in_channel * 2, 2, 2)
        elif sampling == "up":
            self.sampling = nn.Sequential(
                nn.Conv2d(in_channel, in_channel * 2, 1, bias=False), nn.PixelShuffle(2)
            )
        else:
            self.sampling = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.nfbs(x)
        x = self.stn(x)
        x = self.sampling(x)

        return x


class CoarseRestoration(nn.Module):
    def __init__(self):
        super().__init__()

        img_channel = 3
        width = 32

        self.intro = nn.Conv2d(
            in_channels=img_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.outro = nn.Conv2d(
            in_channels=width,
            out_channels=img_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self.encoders = nn.Sequential(
            NAF_STN_Block(width, 128, num_naf=2, sampling="down"),
            NAF_STN_Block(width * 2, 64, num_naf=2, sampling="down"),
            NAF_STN_Block(width * 4, 32, num_naf=4, sampling="down"),
            NAF_STN_Block(width * 8, 16, num_naf=8, sampling="down"),
        )
        self.middle_blocks = NAF_STN_Block(width * 16, 8, num_naf=8)
        self.decoders = nn.Sequential(
            NAF_STN_Block(width * 16, 8, num_naf=2, sampling="up"),
            NAF_STN_Block(width * 8, 16, num_naf=2, sampling="up"),
            NAF_STN_Block(width * 4, 32, num_naf=2, sampling="up"),
            NAF_STN_Block(width * 2, 64, num_naf=2, sampling="up"),
        )

    def forward(self, x: torch.Tensor):
        enc_skips = []

        x = self.intro(x)
        for encoder in self.encoders:
            x = encoder(x)
            enc_skips.append(x)

        x = self.middle_blocks(x)
        for decoder, enc_skip in zip(self.decoders, enc_skips[::-1]):
            x = x + enc_skip
            x = decoder(x)

        x = self.outro(x)

        return x
