import math
from argparse import Namespace

from einops import rearrange
from kornia.augmentation import Normalize, Denormalize
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
        self.mean = [0.507, 0.4865, 0.4409]
        self.std = [0.2673, 0.2564, 0.2761]

        preprocessing = AugmentationSequential(
            Normalize(mean=torch.tensor(self.mean, device=device),
                      std=torch.tensor(self.std, device=device))
        )

        super().__init__(model_path, device, preprocessing)

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
        self.mean = [0.507, 0.4865, 0.4409]
        self.std = [0.2673, 0.2564, 0.2761]

        preprocessing = AugmentationSequential(
            Normalize(mean=torch.tensor(self.mean, device=device),
                      std=torch.tensor(self.std, device=device))
        )

        super().__init__(model_path, device, preprocessing)

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

        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

        preprocessing = AugmentationSequential(
            Normalize(mean=torch.tensor(self.mean, device=device),
                      std=torch.tensor(self.std, device=device))
        )

        super().__init__(model_path, device, preprocessing=preprocessing)

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


class Cifar10NVAEDefenseModel(HLDefenseModel, torch.nn.Module):

    def __init__(self, classifier: BaseClassificationModel, autoencoder_path: str, device: str,
                 resample_from: int, temperature: float = 1.0):
        """
        Defense model using an NVAE pretrained on Cifar10.

        :param resample_from: hierarchy level from which resampling starts.
        :param temperature: temperature for sampling.
        """

        # classifier must be one of the two CNNs pre-trained on Cifar-10
        assert isinstance(classifier, Cifar10VGGModel) or isinstance(classifier, Cifar10ResnetModel), \
            "invalid classifier passed to Cifar10NVAEDefenseModel"

        # no need for preprocessing, since it is done directly in NVAE forward pass.
        super().__init__(classifier, autoencoder_path, device)

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

        # need to reformat codes in the correct shape
        b = codes[0].shape[0]
        d = self.autoencoder.num_latent_per_group
        reshaped_codes = []
        for c in codes:
            r = int(math.sqrt(c.shape[1] // d))
            reshaped_codes.append(c.reshape(b, d, r, r))

        logits = self.autoencoder.decode(reshaped_codes)
        recons = DiscMixLogistic(logits).sample(self.temperature)
        return recons


class CelebAStyleGanDefenseModel(HLDefenseModel, torch.nn.Module):

    def __init__(self, classifier: CelebAResnetModel, autoencoder_path: str, device: str, resample_from: int):
        """
        Defense model using an StyleGan pretrained on FFHQ.
        """

        # takes preprocessed images
        preprocessing = AugmentationSequential(
            Normalize(mean=torch.tensor([0.5, 0.5, 0.5], device=device),
                      std=torch.tensor([0.5, 0.5, 0.5], device=device))
        )
        super().__init__(classifier, autoencoder_path, device, preprocessing)

        self.resample_from = resample_from

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

        b, _, _, _ = batch.shape
        codes = rearrange(codes, 'b n d -> n b d')
        codes = [c for c in codes]

        return codes

    def sample_codes(self, codes: list) -> list:
        """
        HL procedure to re-sampling some codes.
        :param codes: pre-extracted codes as list of tensors, sorted by hierarchy i.
        :return: list of codes of the same shape, where some of them have been re-sampled.
        """
        n_codes = len(codes)
        b, d = codes[0].shape
        device = codes[0].device

        # sample gaussian noise, then get new styles by mixing with codes.
        noises = torch.normal(0, 1, (n_codes, b, d), device=device)
        styles = [self.autoencoder.decoder.style(n) for n in noises]
        styles = codes[:self.resample_from] + styles[self.resample_from:]

        return styles

    def decode(self, codes: list) -> list:
        """
        HL decoding procedure to get images from codes.
        :param codes: pre-extracted codes as list of tensors, sorted by hierarchy i.
        :return: batch of images of shape (B, C, H, W).
        """

        # put codes in the correct shape
        codes = rearrange(torch.stack(codes), 'n b d -> b n d')

        # decode and de-normalize
        reconstructions = self.autoencoder.decode(codes)
        device = reconstructions.device

        reconstructions = Denormalize(
            mean=torch.tensor([0.5, 0.5, 0.5], device=device),
            std=torch.tensor([0.5, 0.5, 0.5], device=device)
        )(reconstructions)

        return reconstructions
