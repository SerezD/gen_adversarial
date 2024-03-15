from abc import ABC, abstractmethod
from kornia.augmentation.container import AugmentationSequential
import torch
from torch import nn

"""
Contains the abstract models for defense.
"""


class BaseClassificationModel(ABC):

    def __init__(self, model_path: str, device: str, preprocessing: AugmentationSequential = None):
        """
        Model containing only a base (pre-trained) classifier, with no additional defenses.
        Useful to compute stuff like clean accuracy or test performances of different attacks.

        :param model_path: absolute path to the pretrained model.
        :param device: cuda device (or cpu) to load the model on
        :param preprocessing: kornia AugmentationSequential object to preprocess each batch before classification.
        """

        super().__init__()

        self.preprocessing = preprocessing.to(device) if preprocessing is not None else None
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
        self.preprocessing = self.preprocessing.to(device) if self.preprocessing is not None else None

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        """
        :param batch: image tensor of shape (B C H W)
        :return un-normalized predictions of shape (B N_CLASSES)
        """

        if self.preprocessing is not None:
            batch = self.preprocessing(batch)

        return self.classifier(batch)


class HLDefenseModel(ABC):

    def __init__(self, classifier: BaseClassificationModel, autoencoder_path: str, device: str,
                 autoencoder_preprocessing: AugmentationSequential = None):
        """
        Model composed of HL-Autoencoder + CNN.
        The autoencoder is used to pre-process the input samples.

        :param classifier: pretrained classification model.
        :param autoencoder_path: absolute path to the pretrained autoencoder.
        :param device: cuda device (or cpu) to load both ae and classifier on
        :param autoencoder_preprocessing: kornia AugmentationSequential object to preprocess each batch before
                                          auto encoding it.
        """

        super().__init__()

        self.classifier = classifier
        self.classifier.set_device(device)

        self.preprocessing = autoencoder_preprocessing.to(device) if autoencoder_preprocessing is not None else None

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
                un-normalized predictions of shape (B, N_CLASSES), computed on the cleaned images.
                else:
                1. un-normalized predictions of shape (B, N_CLASSES), computed on the original images.
                2. un-normalized predictions of shape (B, N_CLASSES), computed on the cleaned images.
                3. the batch of cleaned images (B, C, H, W)
        """

        # preprocessing before autoencoding
        if self.preprocessing is not None:
            batch = self.preprocessing(batch)

        # extract codes (encoding)
        original_codes = self.get_codes(batch)

        # re-sample some codes
        new_codes = self.sample_codes(original_codes)

        # decode
        new_recons = self.decode(new_codes)

        # forward for final classification
        preds_clean = self.classifier(batch)
        preds_new_recons = self.classifier(new_recons)

        if preds_only:
            return preds_new_recons

        return preds_clean, preds_new_recons, new_recons
