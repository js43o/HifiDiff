# https://tutorials.pytorch.kr/intermediate/spatial_transformer_tutorial.html

import torch
from torch import nn
from torch.nn import functional as F


class STNBlock(nn.Module):
    def __init__(self, in_ch, in_res):
        super().__init__()

        self.fc_res = (in_res - 14) / 4  # localization 연산 후 크기 계산

        self.localization = nn.Sequential(
            nn.Conv2d(in_ch, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * self.fc_res * self.fc_res, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2),
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * self.fc_res * self.fc_res)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
