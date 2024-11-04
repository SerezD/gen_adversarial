import torch
from torch import nn
from src.defenses.competitors.nd_vae.modules.models.NVAE_utils import DiscMixLogistic


class NDVaeDefenseModel(nn.Module):

    def __init__(self, base_classifier, nd_vae, noise_std):
        super().__init__()

        self.base_classifier = base_classifier
        self.purifier = nd_vae

        self.noise_std = noise_std

    def purify(self, x):

        # add gaussian noise
        x = x + torch.randn_like(x) * self.noise_std
        x = torch.clamp(x, 0, 1)

        logits = self.purifier(x)[0]
        x_cln = DiscMixLogistic(logits).mean()
        return x_cln

    def forward(self, x):

        x_cln = self.purify(x)
        preds = self.base_classifier(x_cln)
        return preds
