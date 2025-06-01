import torch
from torch import nn
from collections import OrderedDict

from .idc.model import ResNet50
from .denoiser.model import FusedDenoiser
from .fpg.model import FacialPriorGuidance


class FacialRefiner(nn.Module):
    def __init__(self, latent_res=16, idc_ckpt=None, denoiser_ckpt=None):
        super().__init__()

        self.idc = ResNet50()
        self.denoiser = FusedDenoiser(latent_res)
        self.fpg = FacialPriorGuidance()

        if idc_ckpt is not None:
            self.idc.load_state_dict(torch.load(idc_ckpt)["model_state_dict"])
        self.idc.eval()

        if denoiser_ckpt is not None:
            denoiser_weights = torch.load(denoiser_ckpt)["model_state_dict"]
            temp_weights = OrderedDict()
            for k, v in denoiser_weights.items():
                name = k[7:]  # remove `module.`
                temp_weights[name] = v

            denoiser_weights = temp_weights

            self.denoiser.load_state_dict(denoiser_weights, strict=False)
            self.fpg.load_state_dict(denoiser_weights, strict=False)

            for name, param in self.denoiser.named_parameters():
                if name.startswith("intro") or name.startswith("encoders"):
                    # freeze all parameters of the denoiser encoder
                    param.requires_grad = False

    def forward(self, latents, timesteps, cr_face, cr_latent):
        facial_priors = self.fpg(cr_latent)
        identity_embedding = self.idc(cr_face)

        predicted = self.denoiser(latents, timesteps, facial_priors, identity_embedding)

        return predicted
