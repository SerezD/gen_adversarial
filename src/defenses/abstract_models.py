from abc import ABC, abstractmethod
from kornia.enhance import normalize, denormalize
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

        self.mean = mean
        self.std = std
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

    def __init__(self, classifier: BaseClassificationModel, autoencoder_path: str, resample_from: int,
                 device: str, mean: tuple = None, std: tuple = None):
        """
        Model composed of HL-Autoencoder + CNN.
        The autoencoder is used to pre-process the input samples.

        :param classifier: pretrained classification model.
        :param autoencoder_path: absolute path to the pretrained autoencoder.
        :param resample_from: latent scale from where codes start to be resampled for purification.
        :param device: cuda device (or cpu) to load both ae and classifier on
        :param mean: optional param for Normalization and Denormalization operations.
        :param std: optional param for Normalization and Denormalization operations.
        """

        super().__init__()

        self.classifier = classifier
        self.classifier.set_device(device)

        if (mean is not None and std is None) or (mean is None and std is not None):
            raise ValueError("to apply Normalization/Denormalization, please specify both mean and std.")

        self.mean = mean
        self.std = std
        self.preprocess = self.mean is not None
        self.postprocess = self.mean is not None

        self.resample_from = resample_from
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
    def get_codes(self, batch: torch.Tensor) -> list:
        """
        HL encoding procedure to extract the codes.
        :param batch: pre-processed images of shape (B, C, H, W).
        :return: list of codes, sorted by hierarchy i. Each code is a torch tensor of shape (B, N_i)
        """
        pass

    @abstractmethod
    def sample_codes(self, codes: list) -> list:
        """
        HL procedure to re-sampling some codes.
        :param codes: pre-extracted codes as list of tensors, sorted by hierarchy i.
        :return: list of codes of the same shape, where some of them have been re-sampled.
        """
        pass

    @abstractmethod
    def decode(self, codes: list) -> list:
        """
        HL decoding procedure to get images from codes.
        :param codes: pre-extracted codes as list of tensors, sorted by hierarchy i.
        :return: batch of images of shape (B, C, H, W).
        """
        pass

    def __call__(self, batch: torch.Tensor, preds_only: bool = True) -> list:
        """
        :param batch: image tensor of shape (B C H W)
        :return if preds only:
                un-normalized predictions of shape (B, N_CLASSES), computed on the purified images.
                else:
                1. un-normalized predictions of shape (B, N_CLASSES), computed on the purified images.
                2. the batch of purified images (B, C, H, W)
        """

        # preprocessing before autoencoding
        if self.preprocess:
            batch = normalize(batch, torch.tensor(self.mean), torch.tensor(self.std))

        # extract codes (encoding)
        original_codes = self.get_codes(batch)

        # re-sample some codes
        purified_codes = self.sample_codes(original_codes)

        # decode
        purified_recons = self.decode(purified_codes)

        # denormalize before predicting
        if self.postprocess:
            purified_recons = denormalize(purified_recons, torch.tensor(self.mean), torch.tensor(self.std))

        # forward for final classification
        preds_purified = self.classifier(purified_recons)

        if preds_only:
            return preds_purified

        return preds_purified, purified_recons
