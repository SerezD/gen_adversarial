from collections import OrderedDict

from einops import rearrange

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from src.NVAE.mine.custom_ops.inplaced_sync_batchnorm import SyncBatchNormSwish
from src.NVAE.mine.modules import ResidualCellEncoder, EncCombinerCell, NFBlock, ResidualCellDecoder, DecCombinerCell, \
    ARConv2d, Conv2D
from src.NVAE.mine.distributions import Normal


class AutoEncoder(nn.Module):
    def __init__(self, ae_args: dict, resolution: tuple, use_SE: bool = True):
        """
        :param ae_args: hierarchical args
        :param use_SE: use Squeeze and Excitation
        """

        super().__init__()

        # ######################################################################################
        # general params
        self.img_channels, self.image_resolution, _  = resolution
        self.base_channels, self.ch_multiplier = ae_args['initial_channels'], 1
        self.use_SE = use_SE

        # pre and post processing
        self.n_preprocessing_blocks = ae_args['num_pre-post_process_blocks']
        self.n_cells_per_preprocessing_block = ae_args['num_pre-post_process_cells']
        self.n_postprocessing_blocks = ae_args['num_pre-post_process_blocks']
        self.n_cells_per_postprocessing_block = ae_args['num_pre-post_process_cells']

        # logistic mixture
        self.num_mixtures = ae_args['num_logistic_mixtures']

        # encoder - decoder
        self.num_scales = ae_args['num_scales']
        self.groups_per_scale = [
            max(ae_args['min_groups_per_scale'], ae_args['num_groups_per_scale'] // (2 ** i))
            if ae_args['is_adaptive'] else
            ae_args['num_groups_per_scale']
            for i in range(self.num_scales)
        ]  # different on each scale if is_adaptive (scale N has more groups, scale 0 has less)
        self.groups_per_scale.reverse()

        self.num_cells_per_group = ae_args['num_cells_per_group']

        # latent variables and sampling
        self.num_latent_per_group = ae_args['num_latent_per_group']
        self.num_nf_cells = ae_args['num_nf_cells']
        self.use_nf = self.num_nf_cells is not None

        # loss coefficients
        kl_alpha = [(2 ** i) ** 2 / self.groups_per_scale[self.num_scales - i - 1] *
                    torch.ones(self.groups_per_scale[self.num_scales - i - 1])
                    for i in range(self.num_scales)]
        kl_alpha = torch.cat(kl_alpha, dim=0)
        self.kl_alpha = kl_alpha / torch.min(kl_alpha)  # normalize to min 1

        # ######################################################################################
        # initial learned constant
        self.scaling_factor = 2 ** (self.n_preprocessing_blocks + self.num_scales - 1)
        c, r = int(self.scaling_factor * self.base_channels), self.image_resolution // self.scaling_factor
        self.const_prior = nn.Parameter(torch.rand(size=(1, c, r, r)), requires_grad=True)

        # structure
        self.preprocessing_block = self._init_preprocessing()

        self.encoder_tower, self.encoder_combiners, self.encoder_0 = self._init_encoder()

        self.enc_sampler, self.dec_sampler, self.nf_cells = self._init_samplers()

        self.decoder_tower, self.decoder_combiners = self._init_decoder()

        self.postprocessing_block = self._init_postprocessing()

        self.to_logits = self._init_to_logits()

        # ######################################################################################

        # compute values for spectral and batch norm regularization
        self._count_conv_and_batch_norm_layers()

    def _init_preprocessing(self):
        """
        Returns the preprocessing module, structured as:
                init_conv that changes num channels form self.img_channels to self.base_channels
                N blocks of M cells each.
                Each block doubles the number of channels and downsamples H, W by a factor of 2
        """

        preprocessing = OrderedDict()
        preprocessing['init_conv'] = Conv2D(self.img_channels, self.base_channels, kernel_size=3, padding=1,
                                            bias=True, weight_norm=True)

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
            Conv2D(channels, channels, kernel_size=1, bias=True, weight_norm=True),
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
        nf_cells = nn.ModuleList() if self.use_nf else None

        ch_multiplier = self.ch_multiplier

        for s in range(self.num_scales):

            channels = int(self.base_channels * ch_multiplier)
            for g in range(self.groups_per_scale[s]):

                # encoder sampler
                enc_sampler.add_module(f'sampler_{s}:{g}',
                                       Conv2D(channels, 2 * self.num_latent_per_group,
                                              kernel_size=3, padding=1, bias=True, weight_norm=True))

                # build [optional] NF
                this_nf_blocks = []
                for n in range(self.num_nf_cells):
                    this_nf_blocks.append(NFBlock(self.num_latent_per_group))
                nf_cells.add_module(f'nf_{s}:{g}', torch.nn.Sequential(*this_nf_blocks))

                # decoder samplers (first group uses standard gaussian)
                if not (s == 0 and g == 0):
                    dec_sampler.add_module(f'sampler_{s}:{g}',
                                           nn.Sequential(
                                               nn.ELU(),
                                               Conv2D(channels, 2 * self.num_latent_per_group,
                                                         kernel_size=1, padding=0, bias=True, weight_norm=True)
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

                channels = int(self.base_channels * self.ch_multiplier)
                if not is_first:
                    # standard cell
                    cell = ResidualCellDecoder(channels, channels, upsampling=False, use_SE=self.use_SE)
                else:
                    # cell with downsampling (double channels)
                    cell = ResidualCellDecoder(channels, channels // 2, upsampling=True, use_SE=self.use_SE)
                    self.ch_multiplier /= 2

                block[f'cell_{c}'] = cell

            postprocessing[f'block_{b}'] = nn.Sequential(block)

        return nn.Sequential(postprocessing)

    def _init_to_logits(self):

        in_channels = int(self.base_channels * self.ch_multiplier)

        # OUT --> B, N_MIX * 3, C = 3, H, W
        out_channels = int(self.num_mixtures * 3 * self.img_channels)

        to_logits = nn.Sequential(
            nn.ELU(),
            Conv2D(in_channels, out_channels, kernel_size=3, padding=1, bias=True, weight_norm=True)
        )

        return to_logits

    def _count_conv_and_batch_norm_layers(self):
        """
        collect all norm params in Conv2D and gamma param in batch norm for training regularization
        """

        self.all_log_norm = []
        self.all_conv_layers = []
        self.all_bn_layers = []

        for n, layer in self.named_modules():

            if isinstance(layer, Conv2D) or isinstance(layer, ARConv2d):
                self.all_log_norm.append(layer.log_weight_norm)
                self.all_conv_layers.append(layer)

            if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.SyncBatchNorm) or \
                    isinstance(layer, SyncBatchNormSwish):
                self.all_bn_layers.append(layer)

        # left/right singular vectors used for SR
        self.sr_u = {}
        self.sr_v = {}
        self.num_power_iter = 4

    def spectral_norm_parallel(self):
        """
        This method computes spectral normalization for all conv layers in parallel.
        This method should be called after calling the forward method of all the conv layers in each iteration.
        Will be multiplied by λ coefficient
        """

        weights = {}   # a dictionary indexed by the shape of weights
        for l in self.all_conv_layers:
            weight = l.weight_normalized
            weight_mat = weight.view(weight.size(0), -1)
            if weight_mat.shape not in weights:
                weights[weight_mat.shape] = []

            weights[weight_mat.shape].append(weight_mat)

        loss = 0
        for i in weights:
            weights[i] = torch.stack(weights[i], dim=0)
            with torch.no_grad():
                num_iter = self.num_power_iter
                if i not in self.sr_u:
                    num_w, row, col = weights[i].shape
                    self.sr_u[i] = nn.functional.normalize(
                        torch.ones((num_w, row), device=weights[i].device).normal_(0, 1),
                        dim=1, eps=1e-3)

                    self.sr_v[i] = nn.functional.normalize(
                        torch.ones((num_w, col), device=weights[i].device).normal_(0, 1),
                         dim=1, eps=1e-3)

                    # increase the number of iterations for the first time
                    num_iter = 10 * self.num_power_iter

                for j in range(num_iter):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    self.sr_v[i] = nn.functional.normalize(
                        torch.matmul(self.sr_u[i].unsqueeze(1), weights[i]).squeeze(1), dim=1, eps=1e-3)
                    self.sr_u[i] = nn.functional.normalize(
                        torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)).squeeze(2), dim=1, eps=1e-3)

            sigma = torch.matmul(self.sr_u[i].unsqueeze(1), torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)))
            loss += torch.sum(sigma)
        return loss

    def batch_norm_loss(self):
        """
        batch norm regularization term (also multiplied by λ)
        """
        loss = 0
        for l in self.all_bn_layers:
            if l.affine:
                loss += torch.max(torch.abs(l.weight))

        return loss

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
        log_q = dist.log_p(z_0)

        all_q = [dist]
        all_log_q = [log_q]

        # apply normalizing flows
        if self.use_nf:
            z_0 = self.nf_cells.get_submodule('nf_0:0')(z_0)

        # prior p(z_0) is a standard Gaussian (log_sigma 0 --> sigma = 1.)
        dist = Normal(mu=torch.zeros_like(z_0), log_sigma=torch.zeros_like(z_0))
        log_p = dist.log_p(z_0)

        all_p = [dist]
        all_log_p = [log_p]

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
                    log_q = dist.log_p(z_i)

                    all_q.append(dist)
                    all_log_q.append(log_q)

                    # apply NF
                    if self.use_nf:
                        z_i = self.nf_cells.get_submodule(f'nf_{s}:{g}')(z_i)

                    # evaluate log_p(z)
                    dist = Normal(mu_p, log_sig_p)
                    all_p.append(dist)
                    log_p = dist.log_p(z_i)
                    all_log_p.append(log_p)

                    # combine x and z_i
                    x = self.decoder_combiners.get_submodule(f'combiner_{s}:{g}')(x, z_i)

            # upsampling at the end of each scale
            if s < self.num_scales - 1:
                x = scale.get_submodule('upsampling')(x)

        # postprocessing phase
        x = self.postprocessing_block(x)

        # get logits for mixture
        logits = self.to_logits(x)
        logits = rearrange(logits, 'b (n c) h w -> b n c h w', n=self.num_mixtures * 3, c=self.img_channels)

        # compute kl terms TODO (verify what is used and what is not)
        kl_all = []
        kl_diag = []
        log_p_sum, log_q_sum = 0., 0.
        for q, p, log_q, log_p in zip(all_q, all_p, all_log_q, all_log_p):

            if self.use_nf:
                kl_per_var = log_q - log_p
            else:
                kl_per_var = q.kl(p)

            kl_diag.append(torch.mean(torch.sum(kl_per_var, dim=[2, 3]), dim=0))
            kl_all.append(torch.sum(kl_per_var, dim=[1, 2, 3]))
            log_p_sum += torch.sum(log_p, dim=[1, 2, 3])
            log_q_sum += torch.sum(log_q, dim=[1, 2, 3])

        return logits, log_q_sum, log_p_sum, kl_all, kl_diag

    def sample(self, num_samples: int, temperature: float):

        # sample z_0
        samples_shape = (num_samples, self.num_latent_per_group,
                         self.image_resolution // self.scaling_factor, self.image_resolution // self.scaling_factor)
        dist = Normal(mu=torch.zeros(samples_shape).cuda(), log_sigma=torch.zeros(samples_shape).cuda(),
                      temp=temperature)
        z_0, _ = dist.sample()

        # get initial feature
        x = self.const_prior.expand(num_samples, -1, -1, -1)

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

                    # extract params for p (conditioned on previous decoding)
                    mu_p, log_sig_p = torch.chunk(self.dec_sampler.get_submodule(f'sampler_{s}:{g}')(x), 2, dim=1)

                    # sample z_i
                    dist = Normal(mu_p, log_sig_p)
                    z_i, _ = dist.sample()

                    # combine x and z_i
                    x = self.decoder_combiners.get_submodule(f'combiner_{s}:{g}')(x, z_i)

            # upsampling at the end of each scale
            if s < self.num_scales - 1:
                x = scale.get_submodule('upsampling')(x)

        # postprocessing phase
        x = self.postprocessing_block(x)

        # get logits for mixture
        logits = self.to_logits(x)
        logits = rearrange(logits, 'b (n c) h w -> b n c h w', n=self.num_mixtures * 3, c=self.img_channels)

        return logits

    def autoencode(self, gt_images: torch.Tensor):
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

        # apply normalizing flows
        if self.use_nf:
            z_0 = self.nf_cells.get_submodule('nf_0:0')(z_0)

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

                    # apply NF
                    if self.use_nf:
                        z_i = self.nf_cells.get_submodule(f'nf_{s}:{g}')(z_i)

                    # combine x and z_i
                    x = self.decoder_combiners.get_submodule(f'combiner_{s}:{g}')(x, z_i)

            # upsampling at the end of each scale
            if s < self.num_scales - 1:
                x = scale.get_submodule('upsampling')(x)

        # postprocessing phase
        x = self.postprocessing_block(x)

        # get logits for mixture
        logits = self.to_logits(x)
        logits = rearrange(logits, 'b (n c) h w -> b n c h w', n=self.num_mixtures * 3, c=self.img_channels)

        return logits