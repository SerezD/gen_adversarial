from collections import OrderedDict

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from src.NVAE.mine.custom_ops.inplaced_sync_batchnorm import SyncBatchNormSwish
from src.NVAE.mine.distributions import Normal


class SE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        Squeeze and Excitation module, at the end of Residual Cells.
        :param in_channels:
        :param out_channels:
        """

        super().__init__()

        hidden_channels = max(out_channels // 16, 4)

        self.linear_1 = nn.Linear(in_channels, hidden_channels)
        self.linear_2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):

        # gate
        se = torch.mean(x, dim=[2, 3])
        se = se.view(se.size(0), -1)
        se = nn.functional.relu(self.linear_1(se))
        se = nn.functional.sigmoid(self.linear_2(se))
        se = se.view(se.size(0), -1, 1, 1)

        return x * se


class SkipDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        """
        residual skip connection in case of downscaling.
        :param in_channels:
        :param out_channels:
        :param stride:
        """

        super().__init__()

        self.conv_1 = weight_norm(nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=stride))
        self.conv_2 = weight_norm(nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=stride))
        self.conv_3 = weight_norm(nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=stride))
        self.conv_4 = weight_norm(nn.Conv2d(in_channels, out_channels - 3 * (out_channels // 4),
                                            kernel_size=1, stride=stride))

    def forward(self, x):
        out = torch.nn.functional.silu(x)
        conv1 = self.conv_1(out)
        conv2 = self.conv_2(out[:, :, 1:, 1:])
        conv3 = self.conv_3(out[:, :, :, 1:])
        conv4 = self.conv_4(out[:, :, 1:, :])
        out = torch.cat([conv1, conv2, conv3, conv4], dim=1)
        return out


class SkipUp(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int):

        super().__init__()
        self.conv = weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride))

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.conv(x)


class ResidualCellEncoder(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, downsampling: bool, use_SE: bool):
        """
        FIG 4.B of the paper
        :param in_channels:
        :param out_channels:
        :param downsampling:
        :param use_SE:
        """

        super().__init__()

        # skip connection has convs if downscaling
        if downsampling:
            stride = 2
            self.skip_connection = SkipDown(in_channels, out_channels, stride)
        else:
            stride = 1
            self.skip_connection = nn.Identity()

        # (BN - SWISH) + conv 3x3 + (BN - SWISH) + conv 3x3 + SE
        # downsampling in the first conv, depending on stride
        self.residual = nn.Sequential(
            SyncBatchNormSwish(in_channels, eps=1e-5, momentum=0.05),  # using original NVIDIA code for this
            weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)),
            SyncBatchNormSwish(out_channels, eps=1e-5, momentum=0.05),  # using original NVIDIA code for this
            weight_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        )

        if use_SE:
            self.residual.append(SE(out_channels, out_channels))

    def forward(self, x: torch.Tensor):

        residual = 0.1 * self.residual(x)
        x = self.skip_connection(x)

        return x + residual


class ResidualCellDecoder(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, upsampling: bool, use_SE: bool):
        """
        Figure 3a of the paper.

        :param in_channels:
        :param out_channels:
        :param upsampling:
        :param use_SE:
        """

        super().__init__()

        if upsampling:
            self.skip_connection = SkipUp(in_channels, out_channels, stride=1)
        else:
            self.skip_connection = nn.Identity()

        self.use_se = use_SE

        hidden_dim = in_channels * 6

        residual = [nn.UpsamplingNearest2d(scale_factor=2)] if upsampling else []
        residual += [
            nn.SyncBatchNorm(in_channels, eps=1e-5, momentum=0.05),
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            SyncBatchNormSwish(hidden_dim, eps=1e-5, momentum=0.05),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim, bias=False),
            SyncBatchNormSwish(hidden_dim, eps=1e-5, momentum=0.05),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.SyncBatchNorm(out_channels, eps=1e-5, momentum=0.05)
        ]

        if self.use_se:
            residual.append(SE(out_channels, out_channels))

        self.residual = nn.Sequential(*residual)

    def forward(self, x: torch.Tensor):

        residual = 0.1 * self.residual(x)
        x = self.skip_connection(x)

        return x + residual


class EncCombinerCell(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x_enc, x_dec):
        """
        combine encoder ad decoder features
        """

        x_dec = self.conv(x_dec)
        out = x_enc + x_dec
        return out


class DecCombinerCell(nn.Module):
    def __init__(self, feature_channels: int, z_channels: int, out_channels: int):
        """
        combine feature + noise channels during decoding / generation
        """

        super().__init__()

        self.conv = weight_norm(nn.Conv2d(feature_channels + z_channels, out_channels, kernel_size=1))

    def forward(self, x, z):

        out = torch.cat([x, z], dim=1)
        out = self.conv(out)
        return out


class AutoEncoder(nn.Module):
    def __init__(self, ae_args: dict, resolution: tuple, use_SE: bool = True):
        """
        :param ae_args: hierarchical args
        :param use_SE: use Squeeze and Excitation
        """

        super().__init__()

        # ######################################################################################
        # general params
        self.image_resolution, _, self.img_channels = resolution
        self.base_channels, self.ch_multiplier = ae_args['initial_channels'], 1
        self.use_SE = use_SE

        # pre and post processing
        self.n_preprocessing_blocks = ae_args['num_pre-post_process_blocks']
        self.n_cells_per_preprocessing_block = ae_args['num_pre-postprocess_cells']
        self.n_postprocessing_blocks = ae_args['num_pre-post_process_blocks']
        self.n_cells_per_postprocessing_block = ae_args['num_pre-postprocess_cells']

        # encoder - decoder
        self.num_scales = ae_args['num_scales']
        self.groups_per_scale = [
            max(ae_args['min_groups_per_scale'], ae_args['num_groups_per_scale'] // (2**i))
            if ae_args['is_adaptive'] else
            ae_args['num_groups_per_scale']
            for i in range(self.num_scales)
        ]  # different on each scale if is_adaptive (scale N has more groups, scale 0 has less)
        self.groups_per_scale.reverse()

        self.num_cells_per_group = ae_args['num_cells_per_group']

        # latent variables and sampling
        self.num_latent_per_group = ae_args['num_latent_per_group']
        self.num_nf_cells = ae_args['num_nf_cells']

        # ######################################################################################
        # initial learned constant
        scaling_factor = 2 ** (self.n_preprocessing_blocks + self.num_scales - 1)
        c, r = int(scaling_factor * self.base_channels), self.image_resolution // scaling_factor
        self.const_prior = nn.Parameter(torch.rand(size=(1, c, r, r)), requires_grad=True)

        # structure
        self.preprocessing_block = self._init_preprocessing()

        self.encoder_tower, self.encoder_combiners, self.encoder_0 = self._init_encoder()

        self.enc_sampler, self.dec_sampler, self.nf_cells = self._init_samplers()  # TODO nf_cells is not USED

        self.decoder_tower, self.decoder_combiners = self._init_decoder()

        self.postprocessing_block = self._init_postprocessing()


    def _init_preprocessing(self):
        """
        Returns the preprocessing module, structured as:
                init_conv that changes num channels form self.img_channels to self.base_channels
                N blocks of M cells each.
                Each block doubles the number of channels and downsamples H, W by a factor of 2
        """

        preprocessing = OrderedDict()
        preprocessing['init_conv'] = weight_norm(
            nn.Conv2d(self.img_channels, self.base_channels, kernel_size=3, padding=1, bias=True))

        for b in range(self.n_preprocessing_blocks):

            block = OrderedDict()

            for c in range(self.n_cells_per_preprocessing_block):

                is_last = (c == self.n_cells_per_preprocessing_block - 1)

                if not is_last:
                    # standard cell
                    channels = self.base_channels * self.ch_multiplier
                    cell = ResidualCellEncoder(channels, channels, downsampling=False, use_SE=self.use_SE)
                else:
                    # cell with downsampling (double channels)
                    channels = self.base_channels * self.ch_multiplier
                    cell = ResidualCellEncoder(channels, channels * 2, downsampling=True, use_SE=self.use_SE)
                    self.ch_multiplier *= 2

                block[f'cell_{c}'] = cell

            preprocessing[f'block_{b}'] = nn.Sequential(block)
        return nn.Sequential(preprocessing)

    def _init_encoder(self):
        """

        Returns the encoder tower module, the encoder combiner and the encoder 0 module.

        encoder tower has N Scales, indexed from N-1 to 0 (in the sense that N-1 scale is the closest to the input).
        each scale has M groups, where M can optionally change in each scale if the is_adaptive parameter is True.
        each group has K residuals cells, structured as Fig 2b. of the paper

        For each scale > 0, a downsampling operator applies and number of channels is doubled
        For each group (except last one of scale 0) one latent variable will be produced, through the encoder_combiners
        module

        """

        encoder_tower = nn.ModuleList()
        encoder_combiners = nn.ModuleList()

        for s in range(self.num_scales - 1, -1, -1):

            scale = nn.ModuleList()
            channels = int(self.base_channels * self.ch_multiplier)

            # add groups, cells to scale and combiner at the end
            for g in range(self.groups_per_scale[s] - 1, -1, -1):

                group = nn.ModuleList()

                # add cells for this group
                for c in range(self.num_cells_per_group):
                    cell = ResidualCellEncoder(channels, channels, downsampling=False, use_SE=self.use_SE)
                    group.add_module(f'cell_{c}', cell)

                # at the end of each group, except last group and scale, add combiner (for sampling z)
                if not (s == 0 and g == 0):
                    cell = EncCombinerCell(in_channels=channels, out_channels=channels)
                    encoder_combiners.add_module(f'combiner_{s}:{g}', cell)

                # add group to scale
                scale.add_module(f'group_{g}', group)

            # final downsampling (except for last scale)
            if s > 0:

                # cell with downsampling (double channels)
                cell = ResidualCellEncoder(channels, channels * 2, downsampling=True, use_SE=self.use_SE)
                scale.add_module(f'downsampling', cell)
                self.ch_multiplier *= 2

            encoder_tower.add_module(f'scale_{s}', scale)

        # encoder 0 takes the final output of encoder tower
        channels = int(self.base_channels * self.ch_multiplier)
        encoder_0 = nn.Sequential(
            nn.ELU(),
            weight_norm(nn.Conv2d(channels, channels, kernel_size=1, bias=True)),
            nn.ELU())

        return encoder_tower, encoder_combiners, encoder_0

    def _init_samplers(self):
        """
        Returns encoder_sampler, decoder_sampler and optional NF cells for encoders
        encoder sampler (1 per latent variable) takes encoder_combiner output and gives mu, sigma params.
        decoder sampler same but for decoder
        NF cells, if specified, will be used after z_i sampling, only when auto-encoding
        """

        enc_sampler, dec_sampler = nn.ModuleList(), nn.ModuleList()
        nf_cells = nn.ModuleList() if self.num_nf_cells is not None else nn.Identity()

        ch_multiplier = self.ch_multiplier

        for s in range(self.num_scales):

            channels = int(self.base_channels * ch_multiplier)
            for g in range(self.groups_per_scale[s]):

                # encoder sampler
                enc_sampler.add_module(f'sampler_{s}:{g}',
                                       weight_norm(nn.Conv2d(channels, 2 * self.num_latent_per_group,
                                                             kernel_size=3, padding=1, bias=True)))

                # build [optional] NF TODO

                # decoder samplers (first group uses standard gaussian)
                if not (s == 0 and g == 0):
                    dec_sampler.add_module(f'sampler_{s}:{g}',
                                           nn.Sequential(
                                               nn.ELU(),
                                               weight_norm(
                                                    nn.Conv2d(channels, 2 * self.num_latent_per_group,
                                                              kernel_size=1, padding=0, bias=True))
                                            )
                    )

            ch_multiplier /= 2

        return enc_sampler, dec_sampler, nf_cells

    def _init_decoder(self):

        decoder_tower = nn.ModuleList()
        decoder_combiners = nn.ModuleList()

        for s in range(self.num_scales):

            scale = nn.ModuleList()
            channels = int(self.base_channels * self.ch_multiplier)

            for g in range(self.groups_per_scale[s]):

                group = nn.ModuleList()

                if not (s == 0 and g == 0):

                    # add cells for this group
                    for c in range(self.num_cells_per_group):
                        cell = ResidualCellDecoder(channels, channels, upsampling=False, use_SE=self.use_SE)
                        group.add_module(f'cell_{c}', cell)

                    scale.add_module(f'group_{g}', group)

                cell = DecCombinerCell(channels, self.num_latent_per_group, channels)
                decoder_combiners.add_module(f'combiner_{s}:{g}', cell)

            # upsampling cell at scale end (except for last)
            if s < self.num_scales - 1:

                cell = ResidualCellDecoder(channels, channels // 2, upsampling=True, use_SE=self.use_SE)
                scale.add_module(f'upsampling', cell)

                self.ch_multiplier /= 2

            decoder_tower.add_module(f'scale_{s}', scale)

        return decoder_tower, decoder_combiners

    def _init_postprocessing(self):

        postprocessing = OrderedDict()

        for b in range(self.n_postprocessing_blocks):

            block = OrderedDict()

            for c in range(self.n_cells_per_postprocessing_block):

                is_first = (c == 0)

                if not is_first:
                    # standard cell
                    channels = self.base_channels * self.ch_multiplier
                    cell = ResidualCellDecoder(channels, channels, upsampling=False, use_SE=self.use_SE)
                else:
                    # cell with downsampling (double channels)
                    channels = self.base_channels * self.ch_multiplier
                    cell = ResidualCellDecoder(channels, channels // 2, upsampling=True, use_SE=self.use_SE)
                    self.ch_multiplier /= 2

                block[f'cell_{c}'] = cell

            postprocessing[f'block_{b}'] = nn.Sequential(block)

        return nn.Sequential(postprocessing)

    def _init_to_logits(self):

            channels = int(self.base_channels * self.ch_multiplier)
            to_logits = nn.Sequential(
                nn.ELU(),
                weight_norm(nn.Conv2d(channels, 100, kernel_size=3, padding=1, bias=True))
            )

            # DiscMixLogistic(logits, self.num_mix_output, num_bits=self.num_bits)


    def forward(self, gt_images: torch.Tensor):
        """
        :param gt_images: images in 0__1 range
        :return:
        """

        b = gt_images.shape[0]

        # preprocessing phase
        normalized_images = (gt_images * 2) - 1.0  # in range -1 1
        x = self.preprocessing_block(normalized_images)

        # encoding tower phase
        encoder_combiners_x = {}

        for s in range(self.num_scales - 1, -1, -1):

            scale = self.encoder_tower.get_submodule(f'scale_{s}')

            for g in range(self.groups_per_scale[s]):

                group = scale.get_submodule(f'group_{g}')

                for c in range(self.num_cells_per_group):
                    x = group.get_submodule(f'cell_{c}')(x)

                # add intermediate x (will be used as combiner input for this scale)
                if not (s == 0 and g == 0):
                    encoder_combiners_x[f'{s}:{g}'] = x

            if s > 0:
                x = scale.get_submodule(f'downsampling')(x)

        # encoder 0
        x = self.encoder_0(x)

        # obtain q(z_0|x), p(z_0) for KL loss, sample z_0

        # encoder q(z_0|x)
        mu_q, log_sig_q = torch.chunk(self.enc_sampler.get_submodule('sampler_0:0')(x), 2, dim=1)
        dist = Normal(mu_q, log_sig_q)
        z_0, _ = dist.sample()  # uses reparametrization trick
        # log_q_conv = dist.log_p(z_0)  # this is apparently not used anywhere (in the loss computation) TODO

        # apply normalizing flows TODO
        # nf_offset = 0
        # for n in range(self.num_flows):
        #     z, log_det = self.nf_cells[n](z, ftr)
        #     log_q_conv -= log_det
        # nf_offset += self.num_flows

        all_q = [dist]
        # all_log_q = [log_q_conv]

        # prior p(z_0) is a standard Gaussian (log_sigma 0 --> sigma = 1.)
        dist = Normal(mu=torch.zeros_like(z_0), log_sigma=torch.zeros_like(z_0))
        all_p = [dist]

        # log_p_conv = dist.log_p(z) # this is apparently not used anywhere (in the loss computation) TODO
        # all_log_p = [log_p_conv]

        # decoding phase

        # start from constant prior
        x = self.const_prior.expand(b, -1, -1, -1)

        # first combiner (inject z_0)
        x = self.decoder_combiners.get_submodule('combiner_0:0')(x, z_0)

        for s in range(self.num_scales):

            scale = self.decoder_tower.get_submodule(f'scale_{s}')

            for g in range(self.groups_per_scale[s]):

                if not (s == 0 and g == 0):

                    # group forward
                    group = scale.get_submodule(f'group_{g}')

                    for c in range(self.num_cells_per_group):
                        x = group.get_submodule(f'cell_{c}')(x)

                    # obtain q(z_i|x, z_l>i), p(z_i|z_l>i) for KL loss, sample z_i
                    # extract params for p (conditioned on previous decoding)
                    mu_p, log_sig_p = torch.chunk(self.dec_sampler.get_submodule(f'sampler_{s}:{g}')(x), 2, dim=1)

                    # extract params for q (conditioned on encoder features and previous decoding)
                    enc_combiner = self.encoder_combiners.get_submodule(f'combiner_{s}:{g}')
                    enc_sampler = self.enc_sampler.get_submodule(f'sampler_{s}:{g}')
                    mu_q, log_sig_q = torch.chunk(
                        enc_sampler(enc_combiner(encoder_combiners_x[f'{s}:{g}'], x)),
                        2, dim=1)

                    # sample z_i as combination of encoder and decoder params
                    dist = Normal(mu_p + mu_q, log_sig_p + log_sig_q)
                    z_i, _ = dist.sample()
                    # log_q_conv = dist.log_p(z_i)  # apparently not used anywhere (in the loss computation) TODO

                    # # apply NF  TODO
                    # for n in range(self.num_flows):
                    #     z, log_det = self.nf_cells[nf_offset + n](z, ftr)
                    #     log_q_conv -= log_det
                    # nf_offset += self.num_flows
                    # all_log_q.append(log_q_conv)
                    all_q.append(dist)

                    # log_p(z) for evaluation/ loss computation TODO (verify)
                    dist = Normal(mu_p, log_sig_p)
                    all_p.append(dist)
                    # log_p_conv = dist.log_p(z)    # apparently not used TODO
                    # all_log_p.append(log_p_conv)

                    # combine x and z_i
                    x = self.decoder_combiners.get_submodule(f'combiner_{s}:{g}')(x, z_i)

            # upsampling at the end of each scale
            if s < self.num_scales - 1:
                x = scale.get_submodule('upsampling')(x)

        # postprocessing phase
        x = self.postprocessing_block(x)

        # sample image
        logits = self.image_conditional(x)


        return logits


# TODO THIS WILL BE TRAIN

import os
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def get_model_conf(filepath: str):
    import yaml

    # load params
    with open(filepath, 'r', encoding='utf-8') as stream:
        params = yaml.safe_load(stream)

    return params


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, config):

    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = AutoEncoder(config['autoencoder'], config['resolution']).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    img_batch = torch.rand((4, 3, 32, 32), device='cuda:0')
    out = ddp_model(img_batch)

    cleanup()


if __name__ == '__main__':

    mp.spawn(main, args=(1, get_model_conf('conf.yaml'), ))

