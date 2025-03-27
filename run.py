import torch
from models.coarse_restoration.cr_model import CoarseRestoration

model = CoarseRestoration()

X = torch.rand((1, 3, 128, 128))  # (B, C, H, W)

y = model(X)

print(y.shape)
