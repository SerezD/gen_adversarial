from argparse import Namespace

from kornia.augmentation import Normalize
from kornia.augmentation.container import AugmentationSequential
import torch
from torch import nn

from src.classifier.model import ResNet
from src.defenses.abstract_models import BaseClassificationModel, HLDefenseModel
from src.hl_autoencoders.NVAE.mine.distributions import DiscMixLogistic
from src.hl_autoencoders.NVAE.mine.model import AutoEncoder
from src.hl_autoencoders.StyleGan_E4E.psp import pSp

"""
Contains all the models for defense, including BaseClassificationModels and HLDefenseModels
"""


class Cifar10VGGModel(BaseClassificationModel):

    def __init__(self, model_path: str, device: str):
        """
        Wrapper for the Cifar10 Vgg-16 model downloaded from torch hub.
        """

        # takes normalized images.
        preprocessing = AugmentationSequential(
            Normalize(mean=torch.tensor([0.507, 0.4865, 0.4409], device=device),
                      std=torch.tensor([0.2673, 0.2564, 0.2761], device=device))
        )

        super(BaseClassificationModel, self).__init__(model_path, device, preprocessing)

    def load_classifier(self, model_path: str, device: str) -> nn.Module:
        """
        custom method to load the pretrained classifier.
        :param model_path: absolute path to the pretrained model.
        :param device: cuda device (or cpu) to load the model on
        :return: nn.Module of the pretrained classifier
        """
        cnn = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
        cnn = cnn.to(device).eval()
        return cnn


class Cifar10ResnetModel(BaseClassificationModel):

    def __init__(self, model_path: str, device: str):
        """
        Wrapper for the Cifar10 Resnet-32 model downloaded from torch hub.
        """

        # takes normalized images.
        preprocessing = AugmentationSequential(
            Normalize(mean=torch.tensor([0.507, 0.4865, 0.4409], device=device),
                      std=torch.tensor([0.2673, 0.2564, 0.2761], device=device))
        )

        super(BaseClassificationModel, self).__init__(model_path, device, preprocessing)

    def load_classifier(self, model_path: str, device: str) -> nn.Module:
        """
        custom method to load the pretrained classifier.
        :param model_path: absolute path to the pretrained model.
        :param device: cuda device (or cpu) to load the model on
        :return: nn.Module of the pretrained classifier
        """
        cnn = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True)
        cnn = cnn.to(device).eval()
        return cnn


class CelebAResnetModel(BaseClassificationModel):

    def __init__(self, model_path: str, device: str):
        """
        Wrapper for the CelebA-HQ Gender Resnet-50 custom model.
        """

        # no preprocessing needed.
        super(BaseClassificationModel, self).__init__(model_path, device)

    def load_classifier(self, model_path: str, device: str) -> nn.Module:
        """
        custom method to load the pretrained classifier.
        :param model_path: absolute path to the pretrained model.
        :param device: cuda device (or cpu) to load the model on
        :return: nn.Module of the pretrained classifier
        """
        ckpt = torch.load(model_path, map_location='cpu')
        resnet = ResNet(get_weights=False)
        resnet.load_state_dict(ckpt['state_dict'])
        resnet.to(device).eval()
        return resnet


class Cifar10NVAEDefenseModel(HLDefenseModel):

    def __init__(self, classifier: BaseClassificationModel, autoencoder_path: str, device: str,
                 resample_from: int, temperature: float):
        """
        Defense model using an NVAE pretrained on Cifar10.

        :param resample_from: hierarchy level from which resampling starts.
        :param temperature: temperature for sampling.
        """

        # classifier must be one of the two CNNs pre-trained on Cifar-10
        assert isinstance(classifier, Cifar10VGGModel) or isinstance(classifier, Cifar10ResnetModel), \
            "invalid classifier passed to Cifar10NVAEDefenseModel"

        # no need for preprocessing, since it is done directly in NVAE forward pass.
        super(HLDefenseModel, self).__init__(classifier, autoencoder_path, device)

        self.resample_from = resample_from
        self.temperature = temperature

    def load_autoencoder(self, model_path: str, device: str) -> nn.Module:
        """
        custom method to load the pretrained HL autoencoder.
        :param model_path: absolute path to the pretrained model.
        :param device: cuda device (or cpu) to load the model on
        :return: nn.Module of the pretrained autoencoder
        """
        checkpoint = torch.load(model_path, map_location='cpu')

        config = checkpoint['configuration']

        # create model
        nvae = AutoEncoder(config['autoencoder'], config['resolution'])

        nvae.load_state_dict(checkpoint['state_dict'])
        nvae.to(device).eval()
        return nvae

    def get_codes(self, batch: torch.Tensor) -> list:
        """
        HL encoding procedure to extract the codes.
        :param batch: pre-processed images of shape (B, C, H, W).
        :return: list of codes, sorted by hierarchy i. Each code is a torch tensor of shape (B, N_i)
        """
        codes = self.autoencoder.encode(batch, deterministic=True)

        # TODO verify this is needed.
        b, _, _, _ = batch.shape
        codes = [c.view(b, -1) for c in codes]

        return codes

    def sample_codes(self, codes: list) -> list:
        """
        HL procedure to re-sampling some codes.
        :param codes: pre-extracted codes as list of tensors, sorted by hierarchy i.
        :return: list of codes of the same shape, where some of them have been re-sampled.
        """
        return self.autoencoder.resample_codes(codes[:self.resample_from], self.temperature)

    def decode(self, codes: list) -> list:
        """
        HL decoding procedure to get images from codes.
        :param codes: pre-extracted codes as list of tensors, sorted by hierarchy i.
        :return: batch of images of shape (B, C, H, W).
        """

        # TODO, verify chunks are passed in the correct format.
        logits = self.autoencoder.decode(codes)
        recons = DiscMixLogistic(logits).mean()
        return recons


class CelebAStyleGanDefenseModel(HLDefenseModel):

    def __init__(self, classifier: CelebAResnetModel, autoencoder_path: str, device: str):
        """
        Defense model using an StyleGan pretrained on FFHQ.
        """

        # takes preprocessed images
        preprocessing = AugmentationSequential(
            Normalize(mean=torch.tensor([0.5, 0.5, 0.5], device=device),
                      std=torch.tensor([0.5, 0.5, 0.5], device=device))
        )
        super(HLDefenseModel, self).__init__(classifier, autoencoder_path, device, preprocessing)

    def load_autoencoder(self, model_path: str, device: str) -> nn.Module:
        """
        custom method to load the pretrained HL autoencoder.
        :param model_path: absolute path to the pretrained model.
        :param device: cuda device (or cpu) to load the model on
        :return: nn.Module of the pretrained autoencoder
        """
        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts']

        opts['checkpoint_path'] = model_path
        opts['device'] = device
        opts = Namespace(**opts)

        net = pSp(opts)
        net = net.to(device).eval()
        return net

    def get_codes(self, batch: torch.Tensor) -> list:
        """
        HL encoding procedure to extract the codes.
        :param batch: pre-processed images of shape (B, C, H, W).
        :return: list of codes, sorted by hierarchy i. Each code is a torch tensor of shape (B, N_i)
        """
        codes = self.autoencoder.encode(batch)

        # TODO verify this is needed.
        b, _, _, _ = batch.shape
        codes = [c.view(b, -1) for c in codes]

        return codes

    def sample_codes(self, codes: list) -> list:
        """
        HL procedure to re-sampling some codes.
        :param codes: pre-extracted codes as list of tensors, sorted by hierarchy i.
        :return: list of codes of the same shape, where some of them have been re-sampled.
        """
        # TODO need to sample from gaussian of the correct shape, then pass to MLP
        return codes

    def decode(self, codes: list) -> list:
        """
        HL decoding procedure to get images from codes.
        :param codes: pre-extracted codes as list of tensors, sorted by hierarchy i.
        :return: batch of images of shape (B, C, H, W).
        """

        # TODO, verify chunks are passed in the correct format.
        return self.autoencoder.decode(codes)
