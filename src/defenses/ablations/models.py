import math
import torch

from kornia.filters import gaussian_blur2d
from torch import nn

"""
Base Defense Mechanisms adding simple Gaussian Noise or Blurring to input image as purification.
Used for Ablation purposes.
"""


class GaussianNoiseDefenseModel(nn.Module):

    def __init__(self, base_classifier: nn.Module, eps: float = 0.5):
        super().__init__()

        self.base_classifier = base_classifier
        self.eps = eps

    def purify(self, x: torch.Tensor) -> torch.Tensor:

        # Generate Gaussian noise with the same shape as x
        noise = torch.ones_like(x).normal_(0., 1.)

        # Calculate the L2 norm of the noise
        noise_norm = torch.norm(noise.view(noise.size(0), -1), dim=1, keepdim=True)

        # Scale the noise to have the L2 norm equal to eps
        scaled_noise = noise * (self.eps / noise_norm.view(-1, 1, 1, 1))

        # Add the scaled noise to the original image
        return (x + scaled_noise).clamp(0., 1.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_cln = self.purify(x)

        return self.base_classifier(x_cln)


class GaussianBlurDefenseModel(nn.Module):

    def __init__(self, base_classifier: nn.Module):
        super().__init__()

        self.base_classifier = base_classifier

    def purify(self, x: torch.Tensor) -> torch.Tensor:

        b, c, h, w = x.shape

        # resolution = 2^n
        n = math.sqrt(h)

        # kernel_size = 2^(n // 2) - 1
        k = int(2 ** (n // 2) - 1)

        blurred_x = gaussian_blur2d(x, kernel_size=k, sigma=(1., 1.))
        return blurred_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_cln = self.purify(x)

        return self.base_classifier(x_cln)
