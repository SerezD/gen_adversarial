import math
from abc import ABC, abstractmethod
from kornia.enhance import normalize, denormalize
from kornia.filters import gaussian_blur2d
import torch
from torch import nn

"""
Contains the abstract models for defense.
"""


class BaseClassificationModel(ABC):

    def __init__(self, model_path: str, device: str, mean: tuple = None, std: tuple = None):
        """
        Model containing only a base (pre-trained) classifier, with no additional defenses.
        Useful to compute stuff like clean accuracy or test performances of different attacks.

        :param model_path: absolute path to the pretrained model.
        :param device: cuda device (or cpu) to load the model on
        :param mean: optional param for Normalization.
        :param std: optional param for Normalization.
        """

        super().__init__()

        if (mean is not None and std is None) or (mean is None and std is not None):
            raise ValueError("to apply Normalization, please specify both mean and std.")

        self.mean = torch.tensor(mean, device=device) if mean is not None else None
        self.std = torch.tensor(std, device=device) if std is not None else None
        self.preprocess = self.mean is not None

        self.classifier = self.load_classifier(model_path, device)

    @abstractmethod
    def load_classifier(self, model_path: str, device: str) -> nn.Module:
        """
        custom method to load the pretrained classifier.
        :param model_path: absolute path to the pretrained model.
        :param device: cuda device (or cpu) to load the model on
        :return: nn.Module of the pretrained classifier
        """
        pass

    def set_device(self, device: str) -> None:
        """
        :param device: device to move the classifier on
        """
        self.classifier = self.classifier.to(device)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        """
        :param batch: image tensor of shape (B C H W)
        :return un-normalized predictions of shape (B N_CLASSES)
        """

        if self.preprocess:
            batch = normalize(batch, self.mean, self.std)

        return self.classifier(batch)


class HLDefenseModel(ABC):

    def __init__(self, classifier: BaseClassificationModel, autoencoder_path: str,
                 interpolation_alphas: tuple, initial_noise_eps: float = 0.0,
                 apply_gaussian_blur: bool = False, device: str = 'cpu',
                 mean: tuple = None, std: tuple = None):
        """
        Model composed of HL-Autoencoder + CNN.
        The autoencoder is used to pre-process the input samples.

        :param classifier: pretrained classification model.
        :param autoencoder_path: absolute path to the pretrained autoencoder.
        :param interpolation_alphas: express, for each hierarchy, the degree of interpolation between
        reconstruction code (alpha=0) and new sampled code (alpha=1).
        :param initial_noise_eps: l2 bound for optional noise randomly added to images before purification.
        :param apply_gaussian_blur: optional blurring of input images.
        :param device: cuda device (or cpu) to load both ae and classifier on
        :param mean: optional param for Normalization and Denormalization operations.
        :param std: optional param for Normalization and Denormalization operations.
        """

        super().__init__()

        self.eps = initial_noise_eps
        self.blur_input = apply_gaussian_blur

        self.device = device

        self.classifier = classifier
        self.classifier.set_device(device)

        if (mean is not None and std is None) or (mean is None and std is not None):
            raise ValueError("to apply Normalization/Denormalization, please specify both mean and std.")

        self.mean = torch.tensor(mean, device=device) if mean is not None else None
        self.std = torch.tensor(std, device=device) if std is not None else None
        self.preprocess = self.mean is not None
        self.postprocess = self.mean is not None

        self.interpolation_alphas = interpolation_alphas
        self.autoencoder = self.load_autoencoder(autoencoder_path, device)

    @abstractmethod
    def load_autoencoder(self, model_path: str, device: str) -> nn.Module:
        """
        custom method to load the pretrained HL autoencoder.
        :param model_path: absolute path to the pretrained model.
        :param device: cuda device (or cpu) to load the model on
        :return: nn.Module of the pretrained autoencoder
        """
        pass

    @abstractmethod
    def purify(self, batch: torch.Tensor) -> torch.Tensor:
        """
        HL encoding procedure to extract the codes.
        :param batch: pre-processed images of shape (B, C, H, W).
        :return: post_precessed purified reconstructions (B, C, H, W)
        """
        pass

    def add_gaussian_noise(self, x: torch.Tensor):

        # Generate Gaussian noise with the same shape as x
        noise = torch.ones_like(x).normal_(0., 1.)

        # Calculate the L2 norm of the noise
        noise_norm = torch.norm(noise.view(noise.size(0), -1), dim=1, keepdim=True)

        # Scale the noise to have the L2 norm equal to eps
        scaled_noise = noise * (self.eps / noise_norm.view(-1, 1, 1, 1))

        # Add the scaled noise to the original image
        x_noisy = x + scaled_noise

        return x_noisy

    def apply_gaussian_blur(self, x: torch.Tensor):

        if not self.blur_input:
            return x

        b, c, h, w = x.shape

        # resolution = 2^n
        n = math.sqrt(h)

        # kernel_size = 2^(n // 2) - 1
        k = int(2**(n//2) - 1)

        blurred_x = gaussian_blur2d(x, kernel_size=k, sigma=(1., 1.))
        return blurred_x


    def __call__(self, batch: torch.Tensor, preds_only: bool = True) -> list:
        """
        :param batch: image tensor of shape (B C H W)
        :return if preds only:
                un-normalized predictions of shape (B, N_CLASSES), computed on the purified images.
                else:
                1. un-normalized predictions of shape (B, N_CLASSES), computed on the purified images.
                2. the batch of purified images (B, C, H, W)
        """

        # optional preprocessing
        batch = self.apply_gaussian_blur(batch)
        batch = self.add_gaussian_noise(batch)

        # preprocessing before autoencoding
        if self.preprocess:
            batch = normalize(batch, self.mean, self.std)

        # purify
        purified_recons = self.purify(batch)

        # denormalize before predicting
        if self.postprocess:
            purified_recons = denormalize(purified_recons, self.mean, self.std)

        # forward for final classification
        preds_purified = self.classifier(purified_recons)

        if preds_only:
            return preds_purified

        return preds_purified, purified_recons
