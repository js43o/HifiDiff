import torch
from torch.nn import functional as F
from torch import nn
from torchvision.models import vgg19, VGG19_Weights
from piqa import SSIM


class CRLoss(nn.Module):
    def __init__(self, device: torch.device, lambda_pix, lambda_ssim, lambda_vgg):
        super().__init__()
        self.device = device
        self.vgg_features = {}

        self.lambda_pix = lambda_pix
        self.lambda_ssim = lambda_ssim
        self.lambda_vgg = lambda_vgg

        if self.lambda_ssim > 0:
            self.ssim = SSIM().cuda(self.device)

        if self.lambda_vgg > 0:
            vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
            self.vgg = nn.Sequential(*list(vgg.children()))[:22]
            self.vgg.to(device=device)
            self.vgg.eval()

    def calculate_key_region_loss(
        self, pred: torch.Tensor, y: torch.Tensor, y_patch: torch.Tensor
    ):
        loss = 0.0
        for batch in range(y.shape[0]):
            mask = y_patch[batch].sum(dim=0) > 0.0
            mask = mask.unsqueeze(0).expand(y_patch[batch].shape)
            pred_patch = pred[batch] * mask
            gt_patch = y[batch] * mask
            loss += F.smooth_l1_loss(pred_patch, gt_patch)

        return loss / y.shape[0]

    def forward(
        self, pred: torch.Tensor, y: torch.Tensor, y_patch: torch.Tensor = None
    ):
        vgg_loss, ssim_loss = 0.0, 0.0
        if y_patch is not None:
            key_region_loss = self.calculate_key_region_loss(pred, y, y_patch)
        else:
            key_region_loss = 0.0

        if self.lambda_vgg > 0:
            # mean and std of the ImageNet dataset
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

            with torch.no_grad():
                feat_pred = self.vgg((pred - mean) / std)
                feat_y = self.vgg((y - mean) / std)

            vgg_loss = F.smooth_l1_loss(feat_pred, feat_y)

        if self.lambda_ssim > 0:
            ssim_loss = 1 - self.ssim(pred, y)

        return (
            self.lambda_pix * (F.smooth_l1_loss(pred, y) + key_region_loss)
            + self.lambda_ssim * ssim_loss
            + self.lambda_vgg * vgg_loss
        )
