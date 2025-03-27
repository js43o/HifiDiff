import torch
from torch import nn
from .naf import NAFBlock
from .stn import STNBlock


class NAF_STN_Block(nn.Module):
    def __init__(self, in_ch: int, in_res: int, num_naf: int, sampling=None):
        super().__init__()

        self.nfbs = nn.Sequential(*[NAFBlock(in_ch) for _ in range(num_naf)])
        self.stn = STNBlock(in_ch, in_res)
        if sampling == "down":
            self.sampling = nn.AvgPool2d(2)
        elif sampling == "up":
            self.sampling = nn.Upsample(scale_factor=2.0)
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

        self.encoders = nn.Sequential(
            NAF_STN_Block(3, 128, num_naf=2, sampling="down"),
            NAF_STN_Block(3, 64, num_naf=2, sampling="down"),
            NAF_STN_Block(3, 32, num_naf=4, sampling="down"),
            NAF_STN_Block(3, 16, num_naf=8, sampling="down"),
        )
        self.middle_blocks = NAF_STN_Block(3, 8, num_naf=8)
        self.decoders = nn.Sequential(
            NAF_STN_Block(3, 8, num_naf=2, sampling="up"),
            NAF_STN_Block(3, 16, num_naf=2, sampling="up"),
            NAF_STN_Block(3, 32, num_naf=2, sampling="up"),
            NAF_STN_Block(3, 64, num_naf=2, sampling="up"),
        )

    def forward(self, x: torch.Tensor):
        enc_skips = []
        for encoder in self.encoders:
            x = encoder(x)
            enc_skips.append(x)

        x = self.middle_blocks(x)

        for decoder, enc_skip in zip(self.decoders, enc_skips[::-1]):
            x = x + enc_skip
            x = decoder(x)

        return x
