import torch
from models.cr.model import CoarseRestoration

model = CoarseRestoration()

X = torch.rand((1, 3, 128, 128))  # (B, C, H, W)

y = model(X)

print(y.shape)
