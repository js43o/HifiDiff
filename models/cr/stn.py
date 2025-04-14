# https://tutorials.pytorch.kr/intermediate/spatial_transformer_tutorial.html

import torch
from torch import nn
from torch.nn import functional as F
import math


class STNBlock(nn.Module):
    def __init__(self, in_ch, in_res):
        super().__init__()
        
        kernel_sizes = (
            (3, 1)
            if in_res <= 8
            else (5, 3) if in_res <= 16 else (7, 5) if in_res <= 32 else (9, 7)
        )
        fc_res = (
            in_res - kernel_sizes[0] - 2 * kernel_sizes[1] + 3
        ) // 4  # localization 이후 크기 계산
        self.fc_size = 10 * fc_res * fc_res

        self.localization = nn.Sequential(
            nn.Conv2d(in_ch, 8, kernel_size=kernel_sizes[0]),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=kernel_sizes[1]),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(self.fc_size, int(math.sqrt(self.fc_size))),
            nn.ReLU(True),
            nn.Linear(int(math.sqrt(self.fc_size)), 3 * 2),
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.fc_size)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        return x
