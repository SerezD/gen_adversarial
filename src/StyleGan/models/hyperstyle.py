import math
import torch
from torch import nn
import copy
from argparse import Namespace

from src.StyleGan.models.encoders.psp import pSp
from src.StyleGan.models.stylegan2.model import Generator
from src.StyleGan.models.hypernetworks.hypernetwork import SharedWeightsHyperNetResNet, SharedWeightsHyperNetResNetSeparable


class HyperStyle(nn.Module):

    def __init__(self, opts):

        super(HyperStyle, self).__init__()

        self.opts = opts
        self.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2

        # Define architecture
        self.hypernet = self.set_hypernet()
        self.decoder = Generator(self.opts.output_size, 512, 8, channel_multiplier=2)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

        # Load weights if needed
        self.load_weights()

        if self.opts.load_w_encoder:
            self.w_encoder.eval()

    def set_hypernet(self):
        if self.opts.output_size == 1024:
            self.opts.n_hypernet_outputs = 26
        elif self.opts.output_size == 512:
            self.opts.n_hypernet_outputs = 23
        elif self.opts.output_size == 256:
            self.opts.n_hypernet_outputs = 20
        else:
            raise ValueError(f"Invalid Output Size! Support sizes: [1024, 512, 256]!")
        networks = {
            "SharedWeightsHyperNetResNet": SharedWeightsHyperNetResNet(opts=self.opts),
            "SharedWeightsHyperNetResNetSeparable": SharedWeightsHyperNetResNetSeparable(opts=self.opts),
        }
        return networks[self.opts.encoder_type]

    def load_weights(self):

        print(f'Loading HyperStyle from checkpoint: {self.opts.checkpoint_path}')

        ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
        self.hypernet.load_state_dict(self.__get_keys(ckpt, 'hypernet'), strict=True)
        self.decoder.load_state_dict(self.__get_keys(ckpt, 'decoder'), strict=True)
        self.__load_latent_avg(ckpt)

        if self.opts.load_w_encoder:
            self.w_encoder = self.__get_pretrained_w_encoder()

    def forward(self, x, resize=True, input_code=False, randomize_noise=True, return_latents=False,
                return_weight_deltas_and_codes=False, weights_deltas=None, y_hat=None, codes=None):

        if input_code:
            codes = x
        else:
            if y_hat is None:
                assert self.opts.load_w_encoder, "Cannot infer latent code when e4e isn't loaded."
                y_hat, codes = self.get_initial_inversion(x, resize=True)

            # concatenate original input with w-reconstruction or current reconstruction
            x_input = torch.cat([x, y_hat], dim=1)

            # pass through hypernet to get per-layer deltas
            hypernet_outputs = self.hypernet(x_input)
            if weights_deltas is None:
                weights_deltas = hypernet_outputs
            else:
                weights_deltas = [weights_deltas[i] + hypernet_outputs[i] if weights_deltas[i] is not None else None
                                  for i in range(len(hypernet_outputs))]

        input_is_latent = (not input_code)
        images, result_latent = self.decoder([codes],
                                             weights_deltas=weights_deltas,
                                             input_is_latent=input_is_latent,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents)

        if resize:
            images = self.face_pool(images)

        if return_latents and return_weight_deltas_and_codes:
            return images, result_latent, weights_deltas, codes, y_hat
        elif return_latents:
            return images, result_latent
        elif return_weight_deltas_and_codes:
            return images, weights_deltas, codes
        else:
            return images

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None

    @staticmethod
    def __get_keys(d, name):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
        return d_filt

    def __get_pretrained_w_encoder(self):
        print("Loading pretrained W encoder...")
        opts_w_encoder = vars(copy.deepcopy(self.opts))
        opts_w_encoder['checkpoint_path'] = self.opts.w_encoder_checkpoint_path
        opts_w_encoder['encoder_type'] = self.opts.w_encoder_type
        opts_w_encoder['input_nc'] = 3
        opts_w_encoder = Namespace(**opts_w_encoder)
        w_net = pSp(opts_w_encoder)
        w_net = w_net.encoder
        w_net.eval()
        w_net.cuda()
        return w_net

    @torch.no_grad()
    def get_initial_inversion(self, x, resize=True):

        if self.w_encoder.training:
            self.w_encoder.eval()

        codes = self.w_encoder.forward(x)

        if codes.ndim == 2:
            codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        y_hat, _ = self.decoder([codes], weights_deltas=None, input_is_latent=True, randomize_noise=False,
                                return_latents=False)
        if resize:
            y_hat = self.face_pool(y_hat)

        return y_hat, codes

    @torch.no_grad()
    def get_inversion(self, x, delta_weights, resize=True):

        if self.w_encoder.training:
            self.w_encoder.eval()

        codes = self.w_encoder.forward(x)

        if codes.ndim == 2:
            codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        y_hat, _ = self.decoder([codes], weights_deltas=delta_weights, input_is_latent=True, randomize_noise=False,
                                return_latents=False)
        if resize:
            y_hat = self.face_pool(y_hat)

        return y_hat, codes

    @torch.no_grad()
    def decode(self, codes, delta_weights, resize=True):

        y_hat, _ = self.decoder([codes], weights_deltas=delta_weights, input_is_latent=True, randomize_noise=False,
                                return_latents=False)
        if resize:
            y_hat = self.face_pool(y_hat)

        return y_hat
