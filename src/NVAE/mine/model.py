import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from src.NVAE.mine.custom_ops.inplaced_sync_batchnorm import SyncBatchNormSwish


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


class EncCombinerCell(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x1, x2):

        # TODO understand what x1, x2 are
        x2 = self.conv(x2)
        out = x1 + x2
        return out


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

        for b in range(self.n_blocks):
            for c in range(self.n_cells_per_block):
                x = self.preprocessing.get_submodule(f'block_{b}').get_submodule(f'cell_{c}')(x)
        return x


class Encoder(nn.Module):

    def __init__(self, ae_args, mult: int, use_SE: bool):

        super().__init__()

        self.num_scales = ae_args['num_scales']
        self.groups_per_scale = [
            max(ae_args['min_groups_per_scale'], ae_args['num_groups_per_scale'] // 2)
            if ae_args['is_adaptive'] else
            ae_args['num_groups_per_scale']
            for _ in range(self.num_scales)
        ]  # different on each scale if is_adaptive

        self.num_cells_per_group = ae_args['num_cells_per_group']
        self.initial_channels = ae_args['initial_channels']

        self.encoder_tower = nn.ModuleList()

        # Each scale is a downsampling factor
        # Each scale has n groups == latent variable at that resolution/scale
        # Each group has n cells

        for s in range(self.num_scales):

            this_scale = nn.ModuleList()

            # add groups
            for g in range(self.groups_per_scale[s]):

                this_group = nn.ModuleList()

                # add cells for this group
                for c in range(self.num_cells_per_group):

                    channels = int(self.initial_channels * mult)
                    cell = ResidualCellEncoder(channels, channels, downsampling=False, use_SE=use_SE)
                    this_group.add_module(f'cell_{c}', cell)

                # at the end of each group, except last group and scale, add combiner (for sampling z)
                if not (s == self.num_scales - 1 and g == self.groups_per_scale[s] - 1):
                    channels = int(self.initial_channels * mult)
                    cell = EncCombinerCell(in_channels=channels, out_channels=channels)
                    this_group.add_module(f'combiner', cell)

                # in any case, add group to scale
                this_scale.add_module(f'group_{g}', this_group)

            # all groups added, add downsampling (except for last scale)
            if s < self.num_scales - 1:

                # cell with downsampling (double channels)
                channels = int(self.initial_channels * mult)
                cell = ResidualCellEncoder(channels, channels * 2, downsampling=True, use_SE=use_SE)
                mult = mult * 2

                this_scale.add_module('downsampling', cell)

            self.encoder_tower.add_module(f'scale_{s}', this_scale)

    def forward(self, x: torch.Tensor):

        # here combiner cells refer to the + part (combine with decoder for z)
        # TODO, can we remove combiner from here ?
        combiner_cells_enc = []
        combiner_cells_x = []

        for s, scale in enumerate(self.encoder_tower):
            for g in range(self.groups_per_scale[s]):
                for c in range(self.num_cells_per_group):
                    x = scale.get_submodule(f'group_{g}').get_submodule(f'cell_{c}')(x)

                if any('combiner' in name for name, _ in scale.get_submodule(f'group_{g}').named_children()):
                    combiner_cells_enc.append(scale.get_submodule(f'group_{g}').get_submodule(f'combiner'))
                    combiner_cells_x.append(x)

            if any('downsampling' in name for name, _ in scale.named_children()):
                x = scale.get_submodule(f'downsampling')(x)

        # reverse combiner cells and their input for decoder
        combiner_cells_enc.reverse()
        combiner_cells_x.reverse()

        return x, combiner_cells_enc, combiner_cells_x


class AutoEncoder(nn.Module):
    def __init__(self, ae_args: dict, use_SE: bool = True):
        """
        :param ae_args: hierarchical args
        :param use_SE: use Squeeze and Excitation
        """

        super().__init__()

        # structure
        self.preprocess = PreProcessing(img_channels=3, out_channels=ae_args['initial_channels'],
                                        n_blocks=ae_args['num_preprocess_blocks'],
                                        n_cells_per_block=ae_args['num_preprocess_cells'], use_SE=use_SE)

        multiplier = ae_args['num_preprocess_blocks']
        self.encoder = Encoder(ae_args, multiplier, use_SE=use_SE)

    def forward(self, gt_images: torch.Tensor):
        """
        :param gt_images:
        :return:
        """

        # preprocessing phase
        x = self.preprocess(gt_images)

        # encoding phase
        x, combiner_cells_enc, combiner_cells_x = self.encoder(x)

        # decoding phase

        # postprocessing phase
        return x


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
    model = AutoEncoder(config['autoencoder']).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    img_batch = torch.rand((4, 3, 32, 32), device='cuda:0')
    out = ddp_model(img_batch)

    cleanup()


if __name__ == '__main__':

    mp.spawn(main, args=(1, get_model_conf('conf.yaml'), ))

