import torch
from torch import nn
from torch.nn.utils import weight_norm

from NVAE.mine.custom_ops.inplaced_sync_batchnorm import SyncBatchNormSwish


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
            self.skip_connection = SkipDown(in_channels, in_channels * 2, stride)
        else:
            stride = 1
            self.skip_connection = nn.Identity()

        # (BN - SWISH) + conv 3x3 + (BN - SWISH) + conv 3x3 + SE
        # downsampling in the first conv, depending on stride
        self.residual = nn.Sequential([
            SyncBatchNormSwish(in_channels, eps=1e-5, momentum=0.05),  # using original NVIDIA code for this
            weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)),
            SyncBatchNormSwish(out_channels, eps=1e-5, momentum=0.05),  # using original NVIDIA code for this
            weight_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        ])

        if use_SE:
            self.residual.append(SE(out_channels, out_channels))

    def forward(self, x: torch.Tensor):

        residual = 0.1 * self.residual(x)
        x = self.skip_connection(x)

        return x + residual


class ResidualCellDecoder(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, upsampling: bool, use_SE: bool):
        """
        :param in_channels:
        :param out_channels:
        :param upsampling:
        :param use_SE:
        """

        super().__init__()

        if upsampling:
            stride = -1
            skip_connection = SkipUp(in_channels, out_channels // 2, stride)
        else:
            stride = 1
            skip_connection = nn.Identity()

        self.use_se = use_SE
        self._num_nodes = 2
        self._ops = nn.ModuleList()

        # TODO
        # for i in range(self._num_nodes):
        #     i_stride = stride if i == 0 else 1
        #     C = Cin if i == 0 else Cout
        #     primitive = arch[i]
        #     op = OPS[primitive](C, Cout, i_stride)
        #     self._ops.append(op)
        #
        # # SE
        # if self.use_se:
        #     self.se = SE(Cout, Cout)

    def forward(self, s):
        # skip branch
        skip = self.skip(s)
        for i in range(self._num_nodes):
            s = self._ops[i](s)

        s = self.se(s) if self.use_se else s
        return skip + 0.1 * s


class PreProcessing(nn.Module):
    def __init__(self, img_channels: int, out_channels: int, n_blocks: int, n_cells_per_block: int, use_SE: bool):
        """
        take input image and pre-process for encoder input.
        :param img_channels: input image channels (1 if grayscale or 3 if color images)
        :param out_channels: number of initial channels of the encoder
        :param n_blocks: number of preprocessing blocks
        :param n_cells_per_block: number of cells in each preprocessing block
        """

        super().__init__()

        self.img_channels = img_channels
        self.out_channels = out_channels

        self.n_blocks = n_blocks
        self.n_cells_per_block = n_cells_per_block

        # modules
        self.init_conv = weight_norm(nn.Conv2d(img_channels, out_channels, kernel_size=3, padding=1, bias=True))

        self.preprocessing = nn.ModuleList()

        mult = 1
        for b in range(n_blocks):

            block = nn.ModuleList()

            for c in range(n_cells_per_block):

                is_last = (c == n_cells_per_block - 1)

                if not is_last:
                    # standard cell
                    channels = out_channels * mult
                    cell = ResidualCellEncoder(channels, channels, downsampling=False, use_SE=use_SE)
                else:
                    # cell with downsampling (double channels)
                    channels = out_channels * mult
                    cell = ResidualCellEncoder(channels, channels * 2, downsampling=True, use_SE=use_SE)
                    mult = mult * 2

                block.add_module(f'cell_{c}', cell)

            self.preprocessing.add_module(f'block_{b}', block)

    def forward(self, images: torch.Tensor):
        """
        :param images: normalized in 0__1 range
        """

        normalized_images = (images * 2) - 1.0
        x = self.init_conv(normalized_images)

        return self.preprocessing(x)


class Encoder(nn.Module):

    def __init__(self, h_args: dict, e_args: dict, mult: int, use_SE: bool):

        super().__init__()

        self.num_latent_scales = h_args['num_latent_scales']
        self.num_groups_per_scale = h_args['num_groups_per_scale']
        self.num_latent_per_group = h_args['num_latent_per_group']
        self.groups_per_scale = [max(h_args['minimum_groups'], self.num_groups_per_scale // 2) if h_args['is_adaptive']
                                 else self.num_groups_per_scale for _ in range(self.num_latent_scales)]

        self.num_channels_enc = e_args['num_channels_enc']
        self.num_preprocess_blocks = e_args['num_preprocess_blocks']
        self.num_preprocess_cells = e_args['num_preprocess_cells']
        self.num_cell_per_cond_enc = e_args['num_cell_per_cond_enc']

        enc_tower = nn.ModuleList()
        for scale in range(self.num_latent_scales):
            for group in range(self.groups_per_scale[scale]):
                for cell in range(self.num_cell_per_cond_enc):

                    num_c = int(self.num_channels_enc * mult)
                    cell = ResidualCellEncoder(num_c, num_c, downsampling=False, use_SE=use_SE)
                    enc_tower.append(cell)

                # add encoder combiner
                if not (scale == self.num_latent_scales - 1 and group == self.groups_per_scale[scale] - 1):
                    num_ce = int(self.num_channels_enc * mult)
                    num_cd = int(self.num_channels_dec * mult)
                    cell = EncCombinerCell(num_ce, num_cd, num_ce, cell_type='combiner_enc')
                    enc_tower.append(cell)

            # down cells after finishing a scale
            if scale < self.num_latent_scales - 1:
                arch = self.arch_instance['down_enc']
                num_ci = int(self.num_channels_enc * mult)
                num_co = int(CHANNEL_MULT * num_ci)
                cell = Cell(num_ci, num_co, cell_type='down_enc', arch=arch, use_se=self.use_se)
                enc_tower.append(cell)
                mult = CHANNEL_MULT * mult


class AutoEncoder(nn.Module):
    def __init__(self, h_args: dict, e_args: dict, use_SE: bool = True):
        """
        :param h_args: hierarchical args
        :param e_args: encoder args
        :param use_SE: use Squeeze and Excitation
        """

        super().__init__()

        self.use_SE = use_SE  # weather to use the squeeze and excitation module
        # self.writer = writer
        # self.arch_instance = arch_instance
        # self.dataset = args.dataset
        # self.crop_output = self.dataset in {'mnist', 'omniglot', 'stacked_mnist'}
        # self.res_dist = args.res_dist
        # self.num_bits = args.num_x_bits

        # hierarchical params
        self.num_latent_scales = h_args['num_latent_scales']
        self.num_groups_per_scale = h_args['num_groups_per_scale']
        self.num_latent_per_group = h_args['num_latent_per_group']

        self.groups_per_scale = [max(h_args['minimum_groups'], self.num_groups_per_scale // 2) if h_args['is_adaptive']
                                 else self.num_groups_per_scale for _ in range(self.num_latent_scales)]

        # self.vanilla_vae = self.num_latent_scales == 1 and self.num_groups_per_scale == 1

        # encoder params
        self.num_channels_enc = e_args['num_channels_enc']
        self.num_preprocess_blocks = e_args['num_preprocess_blocks']
        self.num_preprocess_cells = e_args['num_preprocess_cells']
        self.num_cell_per_cond_enc = e_args['num_cell_per_cond_enc']

        # # decoder parameters
        # self.num_channels_dec = args.num_channels_dec
        # self.num_postprocess_blocks = args.num_postprocess_blocks
        # self.num_postprocess_cells = args.num_postprocess_cells
        # self.num_cell_per_cond_dec = args.num_cell_per_cond_dec  # number of cell for each conditional in decoder

    def forward(self, gt_images: torch.Tensor):
        """
        :param gt_images:
        :return:
        """

        # preprocessing phase


        # encoding phase

        # decoding phase

        # postprocessing phase
        return gt_images


if __name__ == '__main__':
    def get_model_conf(filepath: str):
        import yaml

        # load params
        with open(filepath, 'r', encoding='utf-8') as stream:
            params = yaml.safe_load(stream)

        return params

    conf = get_model_conf('./conf.yaml')
    test = AutoEncoder(conf['hierarchies'], conf['encoder'])
