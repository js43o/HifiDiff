import torch
from torch import nn

from .naf import NAFBlock


class FacialPriorGuidance(nn.Module):
    def __init__(self):
        super().__init__()

        img_channel = 4
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

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.convs = nn.ModuleList()

        chan = width
        for num in [2, 2, 4, 8]:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.convs.append(
            nn.Sequential(nn.Conv2d(chan, chan, 1, bias=False), nn.PixelShuffle(1))
        )

        for _ in range(4):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)
                )
            )
            chan = chan // 2

    def forward(self, x: torch.Tensor):
        enc_skips = []
        results = []

        x = self.intro(x)
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            enc_skips.append(x)
            x = down(x)

        x = self.convs[0](x)
        results.append(x)

        for conv, enc_skip in zip(self.convs[1:], enc_skips[::-1]):
            x = conv(x)
            x = x + enc_skip
            results.append(x)

        return results
