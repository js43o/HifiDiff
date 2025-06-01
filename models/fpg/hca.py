from torch import nn
import torch.nn.functional as F


class HybridCrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim), nn.Sigmoid()
        )
        self.spatial_mlp = nn.Sequential(
            nn.Conv2d(dim, dim // 2, (1, 1)),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 1, (1, 1)),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        # with padding (3×3 convolution)
        self.fused_mlp = nn.Sequential(
            nn.Conv2d(dim, dim, (3, 3), 1, 1), nn.BatchNorm2d(dim), nn.ReLU()
        )

    def forward(self, f_g, f_d):
        w_c = self.channel_cross_attention(f_g)
        w_s = self.spatial_cross_attention(f_g)
        f_o = f_d + w_c * f_d + w_s * f_d
        f_o = self.fused_mlp(f_o)

        return f_o

    def channel_cross_attention(self, f_g):
        global_avg_pooled = F.adaptive_avg_pool2d(f_g, (1, 1))
        global_max_pooled = F.adaptive_max_pool2d(f_g, (1, 1))
        pooled_sum = global_avg_pooled + global_max_pooled  # (1, 512, 1, 1)

        batch = f_g.shape[0]
        pooled_sum = pooled_sum.reshape((batch, -1))
        w_c = self.channel_mlp(pooled_sum)
        w_c = w_c.reshape(batch, -1, 1, 1)

        return w_c

    def spatial_cross_attention(self, f_g):
        w_s = self.spatial_mlp(f_g)

        return w_s
