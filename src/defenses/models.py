from argparse import Namespace

from einops import rearrange
import torch
from torch import nn

from src.classifier.model import ResNet
from src.defenses.abstract_models import BaseClassificationModel, HLDefenseModel
from src.hl_autoencoders.NVAE.modules.distributions import DiscMixLogistic, Normal
from src.hl_autoencoders.NVAE.model import AutoEncoder
from src.hl_autoencoders.StyleGan_E4E.psp import pSp

"""
Contains all the models for defense, including BaseClassificationModels and HLDefenseModels
"""


class CelebAResnetModel(BaseClassificationModel, torch.nn.Module):

    def __init__(self, model_path: str, device: str):
        """
        Wrapper for the CelebA-HQ Gender Resnet-50 custom model.
        """

        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        super().__init__(model_path, device, mean, std)

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


class NVAEDefenseModel(HLDefenseModel, torch.nn.Module):

    def __init__(self, classifier, autoencoder_path: str,  # TODO classifier
                 interpolation_alphas: tuple, initial_noise_eps: float = 0.0,
                 apply_gaussian_blur: bool = False, device: str = 'cpu', temperature: float = 0.6):
        """
        Defense model using an NVAE.
        :param temperature: temperature for sampling.
        """

        self.temperature = temperature

        # no need for preprocessing, since it is done directly in NVAE forward pass.
        super().__init__(classifier, autoencoder_path, interpolation_alphas, initial_noise_eps,
                         apply_gaussian_blur, device)

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

        nvae.load_state_dict(checkpoint[f'state_dict_temp={self.temperature}'])
        nvae.to(device).eval()
        return nvae

    def purify(self, batch: torch.Tensor) -> torch.Tensor:
        """
        HL encoding procedure to extract the codes.
        :param batch: pre-processed images of shape (B, C, H, W).
        :return: post_precessed purified reconstructions (B, C, H, W)
        """

        b = batch.shape[0]

        # preprocessing phase
        normalized_images = self.autoencoder.normalization(batch)
        x = self.autoencoder.preprocessing_block(normalized_images)

        # encoding tower phase
        encoder_combiners_x = {}

        for s in range(self.autoencoder.num_scales - 1, -1, -1):

            scale = self.autoencoder.encoder_tower.get_submodule(f'scale_{s}')

            for g in range(self.autoencoder.groups_per_scale[s]):

                group = scale.get_submodule(f'group_{g}')

                for c in range(self.autoencoder.num_cells_per_group):
                    x = group.get_submodule(f'cell_{c}')(x)

                # add intermediate x (will be used as combiner input for this scale)
                if not (s == 0 and g == 0):
                    encoder_combiners_x[f'{s}:{g}'] = x

            if s > 0:
                x = scale.get_submodule(f'downsampling')(x)

        # encoder 0
        x = self.autoencoder.encoder_0(x)

        # encoder q(z_0|x)
        mu_q, log_sig_q = torch.chunk(self.autoencoder.enc_sampler.get_submodule('sampler_0:0')(x), 2, dim=1)
        dist_enc = Normal(mu_q, log_sig_q)

        # decoder p(z_0)
        dist_dec = Normal(mu=torch.zeros_like(mu_q),
                          log_sigma=torch.zeros_like(log_sig_q),
                          temp=self.temperature)

        z_0 = (1 - self.interpolation_alphas[0]) * dist_enc.mu + self.interpolation_alphas[0] * dist_dec.sample()[0]

        # apply normalizing flows
        if self.autoencoder.use_nf:
            z_0 = self.autoencoder.nf_cells.get_submodule('nf_0:0')(z_0)

        # decoding phase

        # start from constant prior
        x = self.autoencoder.const_prior.expand(b, -1, -1, -1)

        # first combiner (inject z_0)
        x = self.autoencoder.decoder_combiners.get_submodule('combiner_0:0')(x, z_0)

        latent_idx = 1

        for s in range(self.autoencoder.num_scales):

            scale = self.autoencoder.decoder_tower.get_submodule(f'scale_{s}')

            for g in range(self.autoencoder.groups_per_scale[s]):

                if not (s == 0 and g == 0):

                    # group forward
                    group = scale.get_submodule(f'group_{g}')

                    for c in range(self.autoencoder.num_cells_per_group):
                        x = group.get_submodule(f'cell_{c}')(x)

                    # obtain z_i interpolating encoder q(z_i|x, z<i) and decoder p(z_i|z<i)
                    enc_combiner = self.autoencoder.encoder_combiners.get_submodule(f'combiner_{s}:{g}')
                    enc_sampler = self.autoencoder.enc_sampler.get_submodule(f'sampler_{s}:{g}')
                    mu_q, log_sig_q = torch.chunk(
                        enc_sampler(enc_combiner(encoder_combiners_x[f'{s}:{g}'], x)),
                        2, dim=1)

                    dec_sampler = self.autoencoder.dec_sampler.get_submodule(f'sampler_{s}:{g}')
                    mu_p, log_sig_p = torch.chunk(dec_sampler(x), 2, dim=1)

                    dist_enc = Normal(mu_p + mu_q, log_sig_p + log_sig_q)
                    dist_dec = Normal(mu_p, log_sig_p, temp=self.temperature)

                    z_i = (1 - self.interpolation_alphas[latent_idx]) * dist_enc.mu
                    z_i += self.interpolation_alphas[latent_idx] * dist_dec.sample()[0]

                    # apply NF
                    if self.autoencoder.use_nf:
                        z_i = self.autoencoder.nf_cells.get_submodule(f'nf_{s}:{g}')(z_i)

                    # combine x and z_i
                    x = self.autoencoder.decoder_combiners.get_submodule(f'combiner_{s}:{g}')(x, z_i)

                    latent_idx += 1

            # upsampling at the end of each scale
            if s < self.autoencoder.num_scales - 1:
                x = scale.get_submodule('upsampling')(x)

        # postprocessing phase
        x = self.autoencoder.postprocessing_block(x)

        # get logits for mixture
        logits = self.autoencoder.to_logits(x)

        disc_mix = DiscMixLogistic(logits, img_channels=3, num_bits=8)
        reconstructions = disc_mix.sample()

        return self.autoencoder.denormalization(reconstructions)


class CelebAStyleGanDefenseModel(HLDefenseModel, torch.nn.Module):

    def __init__(self, classifier: CelebAResnetModel, autoencoder_path: str,
                 interpolation_alphas: tuple,
                 initial_noise_eps: float = 0.0, apply_gaussian_blur: bool = False,
                 device: str = 'cpu'):
        """
        Defense model using an StyleGan pretrained on FFHQ.
        """

        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

        super().__init__(classifier, autoencoder_path, interpolation_alphas, initial_noise_eps, apply_gaussian_blur,
                         device, mean, std)

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

    def purify(self, batch: torch.Tensor) -> torch.Tensor:
        """
        HL encoding procedure to extract the codes.
        :param batch: pre-processed images of shape (B, C, H, W).
        :return: post_precessed purified reconstructions (B, C, H, W)
        """
        codes = self.autoencoder.encode(batch)

        b, n_codes, d = codes.shape
        device = codes.device

        codes = rearrange(codes, 'b n d -> n b d')

        # sample gaussian noise, then get new styles by mixing with codes.
        noises = torch.normal(0, 1, (n_codes, b, d), device=device)
        styles = torch.stack([self.autoencoder.decoder.style(n) for n in noises], dim=0)

        # interpolate codes and styles
        alphas = torch.tensor(self.interpolation_alphas, device=device).view(-1, 1, 1)
        codes = (1 - alphas) * codes + alphas * styles

        # put codes in the correct shape
        codes = rearrange(codes, 'n b d -> b n d')

        # decode and de-normalize
        reconstructions = self.autoencoder.decode(codes)

        return reconstructions
