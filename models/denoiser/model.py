import torch
from torch import nn
from models.common.naf import NAFBlock


class Denoiser(nn.Module):
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
        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=img_channel,
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
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(8)])

        for num in [2, 2, 2, 2]:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

    def forward(self, input: torch.Tensor):
        B, C, H, W = input.shape

        x = self.intro(input)

        enc_skips = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            enc_skips.append(x)
            x = down(x)

        for middle_blk in self.middle_blks:
            x = middle_blk(x)
            # TODO: element-wise addition with output of IDC

        for decoder, up, enc_skip in zip(self.decoders, self.ups, enc_skips[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + input

        return x[:, :, :H, :W]
