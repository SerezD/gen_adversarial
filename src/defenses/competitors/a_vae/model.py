import torch
from torch import nn
from torch.autograd import Variable

from src.defenses.competitors.a_vae.modules import EncodeConvBlock, StyledConvBlock, EqualConv2d, PixelNorm, \
    EqualLinear, ConvBlock


class Encoder(nn.Module):
    def __init__(self, input_channel, out_channel):

        super(Encoder, self).__init__()

        self.out_channel = out_channel

        self.conv2 = EncodeConvBlock(input_channel, out_channel // 2)  # 32*32*128->16*16*256
        self.conv3 = EncodeConvBlock(out_channel // 2, out_channel)  # 16*16*256->8*8*512
        self.conv4 = EncodeConvBlock(out_channel, 2 * out_channel, norm=False)  # 8*8*512->4*4*512

    def forward(self, x):
        x1 = self.conv2(x)  # skip connection!
        x = self.conv3(x1)
        x = self.conv4(x)
        mu = x[:, :self.out_channel, :, :]
        var = x[:, self.out_channel:, :, :]
        return x1, mu, var


class Generator(nn.Module):
    def __init__(self, output_size: int, fused: bool = True):

        super().__init__()

        # changed this to allow resolution 64, 128, 256
        # note: includes skip connection from encoder
        if output_size == 64:
            convs = [
                StyledConvBlock(512, 512, 3, 1, initial=True),  # 4
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 8
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 16
                StyledConvBlock(512 + 256, 256, 3, 1, upsample=True, fused=fused),  # 32
                StyledConvBlock(256, 128, 3, 1, upsample=True, fused=fused),  # 64
            ]

        elif output_size == 128:
            convs = [
                    StyledConvBlock(512, 512, 3, 1, initial=True),  # 4
                    StyledConvBlock(512, 512, 3, 1, upsample=True),  # 8
                    StyledConvBlock(512, 512, 3, 1, upsample=True),  # 16
                    StyledConvBlock(512 + 256, 256, 3, 1, upsample=True, fused=fused),  # 32
                    StyledConvBlock(256, 256, 3, 1, upsample=True, fused=fused),  # 64
                    StyledConvBlock(256, 128, 3, 1, upsample=True, fused=fused),  # 128
            ]
        elif output_size == 256:
            convs = [
                StyledConvBlock(512, 512, 3, 1, initial=True),  # 4
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 8
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 16
                StyledConvBlock(512 + 256, 256, 3, 1, upsample=True, fused=fused),  # 32
                StyledConvBlock(256, 256, 3, 1, upsample=True, fused=fused),  # 64
                StyledConvBlock(256, 256, 3, 1, upsample=True, fused=fused),  # 128
                StyledConvBlock(256, 128, 3, 1, upsample=True, fused=fused),  # 256
            ]
        else:
            raise NotImplementedError(f"Output size {output_size} is not supported")

        self.progression = nn.ModuleList(
            convs
        )

        self.to_rgb = EqualConv2d(128, 3, 1)

    def forward(self, x_skip, m, v, get_style, noise, inference=False):

        if inference:
            temp = 0.6
        else:
            temp = 1.

        # sample
        v = torch.exp(v * 0.5) * temp
        eps = Variable(torch.randn_like(m), requires_grad=True)
        out = m + eps * v

        z = [out.view(len(out), -1)]

        # get styles
        style = [get_style(z_i) for z_i in z]
        inject_index = [len(self.progression) + 1]

        crossover = 0
        for i, conv in enumerate(self.progression):
            if crossover < len(inject_index) and i > inject_index[crossover]:
                crossover = min(crossover + 1, len(style))

            style_step = style[crossover]

            if out.size(2) == x_skip.size(2):
                out = torch.cat((out, x_skip), dim=1)

            out = conv(out, style_step, noise[i])

        out = self.to_rgb(out)

        return out


class StyledGenerator(nn.Module):
    def __init__(self, output_size: int = 128):

        super().__init__()

        self.encoder = Encoder(3, 512)
        self.generator = Generator(output_size)

        # styles
        n_mlp = 3
        layers = [PixelNorm()]
        layers.append(EqualLinear(512 * 4 * 4, 512))
        layers.append(nn.LeakyReLU(0.2))
        for i in range(n_mlp):
            layers.append(EqualLinear(512, 512))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)

    def forward(self, input, noise=None, inference: bool = False):

        if noise is None:
            batch = len(input)
            noise = []
            for i in range(0, len(self.generator.progression)):
                size = 4 * 2 ** i
                noise.append(torch.randn(batch, 1, size, size).cuda())

        x, m, v = self.encoder(input)

        if inference:
            return self.generator(x, m, v, self.style, noise, inference)

        return m, v, self.generator(x, m, v, self.style, noise)


class Discriminator(nn.Module):
    def __init__(self, initial_res: int = 128, fused=True):

        super().__init__()

        if initial_res == 64:
            convs = [
                ConvBlock(64, 128, 3, 1, downsample=True, fused=fused),  # 64
                ConvBlock(128, 256, 3, 1, downsample=True, fused=fused),  # 32
                ConvBlock(256, 512, 3, 1, downsample=True),  # 16
                ConvBlock(512, 512, 3, 1, downsample=True),  # 8
                ConvBlock(512, 512, 3, 1, 4, 0, norm=False),
            ]

        elif initial_res == 128:
            convs = [
                ConvBlock(64, 128, 3, 1, downsample=True, fused=fused),  # 128
                ConvBlock(128, 256, 3, 1, downsample=True, fused=fused),  # 64
                ConvBlock(256, 512, 3, 1, downsample=True),  # 32
                ConvBlock(512, 512, 3, 1, downsample=True),  # 16
                ConvBlock(512, 512, 3, 1, downsample=True),  # 8
                ConvBlock(512, 512, 3, 1, 4, 0, norm=False),
            ]
        elif initial_res == 256:
            convs = [
                ConvBlock(64, 128, 3, 1, downsample=True, fused=fused),  # 256
                ConvBlock(128, 256, 3, 1, downsample=True, fused=fused),  # 128
                ConvBlock(256, 256, 3, 1, downsample=True),  # 64
                ConvBlock(256, 512, 3, 1, downsample=True),  # 32
                ConvBlock(512, 512, 3, 1, downsample=True),  # 16
                ConvBlock(512, 512, 3, 1, downsample=True),  # 8
                ConvBlock(512, 512, 3, 1, 4, 0, norm=False),
            ]
        else:
            raise NotImplementedError(f"Initial resolution {initial_res} is not supported")

        self.from_rgb = EqualConv2d(3, 64, 1)
        self.progression = nn.Sequential(*convs)
        self.linear = EqualLinear(512, 1)

    def forward(self, input):

        out = self.from_rgb(input)
        out = self.progression(out)
        out = out.squeeze(2).squeeze(2)
        out = self.linear(out)

        return out
