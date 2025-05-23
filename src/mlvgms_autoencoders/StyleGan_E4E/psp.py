import torch
from torch import nn

from src.mlvgms_autoencoders.StyleGan_E4E.encoding.encoder import Encoder4Editing
from src.mlvgms_autoencoders.StyleGan_E4E.stylegan2.generator import Generator


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class pSp(nn.Module):

    def __init__(self, opts):

        super().__init__()

        self.opts = opts

        # Define architecture
        self.encoder = self.set_encoder()
        self.decoder = Generator(opts.stylegan_size, 512, 8, channel_multiplier=2)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

        # Load weights if needed
        self.load_weights()

    def set_encoder(self):

        assert self.opts.encoder_type == 'Encoder4Editing'

        encoder = Encoder4Editing(50, 'ir_se', self.opts)

        return encoder

    def load_weights(self):

        print('Loading e4e over the pSp framework from checkpoint: {}'.format(self.opts.checkpoint_path))
        ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
        self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
        self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
        self.__load_latent_avg(ckpt)

    def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None, is_cars=False):

        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            # normalize with respect to the center of an average face
            if self.opts.start_from_latent_avg:
                if codes.ndim == 2:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
                else:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

            if codes.shape[1] == 18 and is_cars:
                codes = codes[:, :16, :]

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        input_is_latent = not input_code
        images, result_latent = self.decoder([codes],
                                             input_is_latent=input_is_latent,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images

    def encode(self, x: torch.Tensor):
        """
        takes batch of images and returns tensor of codes (B, NLATENTS, DIM)
        """
        codes = self.encoder(x)

        # normalize with respect to the center of an average face
        if self.opts.start_from_latent_avg:
            if codes.ndim == 2:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
            else:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        return codes

    def get_codes(self, x: torch.Tensor):

        codes = self.encode(x)
        codes = [c for c in codes.permute(1, 0, 2)]
        return codes

    def decode(self, codes: torch.Tensor):
        """
        get latents of shape B, NLATENTS, DIM and decode images
        """
        images, _ = self.decoder([codes], input_is_latent=True, randomize_noise=False)
        images = self.face_pool(images)
        return images

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
        elif self.opts.start_from_latent_avg:
            # Compute mean code based on a large number of latents (10,000 here)
            with torch.no_grad():
                self.latent_avg = self.decoder.mean_latent(10000).to(self.opts.device)
        else:
            self.latent_avg = None
        if repeat is not None and self.latent_avg is not None:
            self.latent_avg = self.latent_avg.repeat(repeat, 1)
