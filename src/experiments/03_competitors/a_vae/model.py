import torch

from torch import nn
from torch.nn import init
from torch.nn import functional
from torch.autograd import Function
from torch.autograd import Variable
from math import sqrt

import random


def init_linear(linear):
    init.kaiming_normal_(linear.weight)
    # init.xavier_normal_(linear.weight, gain=0.02)
    linear.bias.data.zero_()


def init_conv(conv):
    init.kaiming_normal_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, _):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class FusedUpsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, x):
        weight = functional.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = functional.conv_transpose2d(x, weight, self.bias, stride=2, padding=self.pad)

        return out


class FusedDownsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, x):
        weight = functional.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = functional.conv2d(x, weight, self.bias, stride=2, padding=self.pad)

        return out


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = functional.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = functional.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None


class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = functional.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None


blur = BlurFunction.apply


class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)
        # return F.conv2d(input, self.weight, padding=1, groups=input.shape[1])


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        # init_conv(self.conv)
        conv.weight.data.normal_()
        # conv.weight.data.xavier_normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        # init_linear(self.linear)
        # linear.weight.data.xavier_normal_()
        linear.bias.data.zero_()
        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        padding,
        kernel_size2=None,
        padding2=None,
        downsample=False,
        fused=False,
        norm=True,
    ):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        if norm == True:
            self.conv1 = nn.Sequential(
                EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
                nn.InstanceNorm2d(num_features=out_channel),
                nn.LeakyReLU(0.2),
            )

            if downsample:
                if fused:
                    self.conv2 = nn.Sequential(
                        Blur(out_channel),
                        FusedDownsample(out_channel, out_channel, kernel2, padding=pad2),
                        nn.InstanceNorm2d(num_features=out_channel),
                        nn.LeakyReLU(0.2),
                    )

                else:
                    self.conv2 = nn.Sequential(
                        Blur(out_channel),
                        EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                        nn.AvgPool2d(2),
                        nn.InstanceNorm2d(num_features=out_channel),
                        nn.LeakyReLU(0.2),
                    )

            else:
                self.conv2 = nn.Sequential(
                    EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                    nn.InstanceNorm2d(num_features=out_channel),
                    nn.LeakyReLU(0.2),
                )
        else:
            self.conv1 = nn.Sequential(
                EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
                nn.LeakyReLU(0.2),
            )

            if downsample:
                if fused:
                    self.conv2 = nn.Sequential(
                        Blur(out_channel),
                        FusedDownsample(out_channel, out_channel, kernel2, padding=pad2),
                        nn.LeakyReLU(0.2),
                    )

                else:
                    self.conv2 = nn.Sequential(
                        Blur(out_channel),
                        EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                        nn.AvgPool2d(2),
                        nn.LeakyReLU(0.2),
                    )

            else:
                self.conv2 = nn.Sequential(
                    EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                    nn.LeakyReLU(0.2),
                )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        return out


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        style_dim=512,
        initial=False,
        upsample=False,
        fused=False,
    ):
        super().__init__()

        if initial:
            self.conv1 = ConstantInput(in_channel)

        else:
            if upsample:
                if fused:
                    self.conv1 = nn.Sequential(
                        FusedUpsample(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )

                else:
                    self.conv1 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        EqualConv2d(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )

            else:
                self.conv1 = EqualConv2d(
                    in_channel, out_channel, kernel_size, padding=padding
                )

        self.noise1 = equal_lr(NoiseInjection(out_channel))
        # self.norm1 = nn.InstanceNorm2d(out_channel)
        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.noise2 = equal_lr(NoiseInjection(out_channel))
        # self.norm2 = nn.InstanceNorm2d(out_channel)
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input, style, noise):
        out = self.conv1(input)
        out = self.noise1(out, noise)
        out = self.lrelu1(out)
        # out = self.norm1(out)
        out = self.adain1(out, style)

        out = self.conv2(out)
        out = self.noise2(out, noise)
        out = self.lrelu2(out)
        # out = self.norm2(out)
        out = self.adain2(out, style)

        return out

class EncodeConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
    ):
        super().__init__()

        self.conv1 = EqualConv2d(
            in_channel, out_channel, kernel_size, stride=1, padding=padding
        )
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.norm1 = nn.InstanceNorm2d(out_channel)

        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, stride=2, padding=padding)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.norm2 = nn.InstanceNorm2d(out_channel)

    def forward(self, input):
        out = self.conv1(input)
        self.norm1(out)
        out = self.lrelu1(out)

        out = self.conv2(out)
        self.norm2(out)
        out = self.lrelu2(out)

        return out

class EncodeConvBlockN(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
    ):
        super().__init__()

        self.conv1 = EqualConv2d(
            in_channel, out_channel, kernel_size, stride=1, padding=padding
        )
        self.noise1 = equal_lr(NoiseInjection(out_channel))
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.norm1 = nn.InstanceNorm2d(out_channel)

        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, stride=2, padding=padding)
        self.noise2 = equal_lr(NoiseInjection(out_channel))
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.norm2 = nn.InstanceNorm2d(out_channel)


    def forward(self, input, noise):
        out = self.conv1(input)
        out = self.noise1(out, noise[0])
        self.norm1(out)
        out = self.lrelu1(out)


        out = self.conv2(out)
        out = self.noise2(out, noise[1])
        self.norm2(out)
        out = self.lrelu2(out)

        return out

class encoder(nn.Module):
    def __init__(self, input_channel, out_channel):
        super(encoder, self).__init__()
        self.out_channel = out_channel
        # self.conv0 = EncodeConvBlock(input_channel, int(out_channel / 2))  # 128*128*3 -> 64*64*128
        # self.conv1 = EncodeConvBlock(int(out_channel/2), int(out_channel/2))  # 64*64*128 -> 32*32*128
        # self.conv2 = EncodeConvBlock(int(out_channel/2), int(out_channel/2))  # 32*32*128->16*16*256
        self.conv3 = EncodeConvBlock(input_channel, out_channel)  # 16*16*256->8*8*512
        self.conv4 = EncodeConvBlock(out_channel, 2 * out_channel)  # 8*8*512->4*4*512
        #self.avgpool = nn.AvgPool2d(4, stride=1)
        # self.fc = nn.Sequential(nn.Linear(4*4*512, 512),
        #                         nn.BatchNorm1d(num_features=512),
        #                         nn.LeakyReLU(0.2))
        # two linear to get the mu vector and the diagonal of the log_variance
        # self.l_mu = nn.Linear(512, 512)
        # self.l_var = nn.Linear(512, 512)

    def forward(self, x):
        # x = self.conv0(x)
        # x = self.conv1(x)
        # x1 = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        mu = x[:, :self.out_channel, :, :]
        var = x[:, self.out_channel:, :, :]
        # x = self.avgpool(x)
        # x = x.view(len(x), -1)
        # x = self.fc(x)
        # mu = self.l_mu(x)
        # var = self.l_var(x)

        return mu, var    # [out2, out3, out4] #[inter_out, out]


class Generator(nn.Module):
    def __init__(self, fused=True, output_size: int = 128):
        super().__init__()

        convs = [
            StyledConvBlock(512, 512, 3, 1, initial=True),  # 4
            StyledConvBlock(512, 512, 3, 1, upsample=True),  # 8
            StyledConvBlock(512, 512, 3, 1, upsample=True),  # 16
            StyledConvBlock(512, 256, 3, 1, upsample=True, fused=fused),  # 32
            StyledConvBlock(256, 256, 3, 1, upsample=True, fused=fused),  # 64
            StyledConvBlock(256, 128, 3, 1, upsample=True, fused=fused),  # 128
        ]
        if output_size == 256:
            convs += [
                StyledConvBlock(128, 64, 3, 1, upsample=True, fused=fused),
            ]


        # StyledConvBlock(64, 32, 3, 1, upsample=True, fused=fused),  # 512
        # StyledConvBlock(32, 16, 3, 1, upsample=True, fused=fused),  # 1024

        self.progression = nn.ModuleList(
            convs
        )

        self.to_rgb = EqualConv2d(128 if output_size < 256 else 64, 3, 1)
        # self.fc = nn.Sequential(
        #     nn.InstanceNorm2d(num_features=512),
        #     nn.LeakyReLU(0.2))
        # self.blur = Blur()

    def forward(self, stylem, m, v, noise, mixing_range=(-1, -1)):
        variances = torch.exp(v * 0.5)
        # sample from a gaussian
        ten_from_normal = Variable(torch.randn_like(m), requires_grad=True)
        # shift and scale using the means and variances

        out = ten_from_normal * variances + m
        z = out.view(len(out), -1)
        style = []
        # step = 5
        if type(z) not in (list, tuple):
            z = [z]
        for i in z:
            style.append(stylem(i))

        # out = self.fc(ten)
        # out = ten.view(len(ten), -1, 4, 4)
        # out = ten #[-1]
        if len(style) < 2:
            inject_index = [len(self.progression) + 1]

        else:
            inject_index = random.sample(list(range(5)), len(style) - 1)

        crossover = 0
        for i, conv in enumerate(self.progression):
            if mixing_range == (-1, -1):
                if crossover < len(inject_index) and i > inject_index[crossover]:
                    crossover = min(crossover + 1, len(style))

                style_step = style[crossover]

            else:
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = style[1]

                else:
                    style_step = style[0]

            out = conv(out, style_step, noise[i])

            if i == len(self.progression) - 1:
                out = self.to_rgb(out)

        return out


class StyledGenerator(nn.Module):
    def __init__(self, code_dim=512, input_size: int = 16, output_size: int = 128):
        super().__init__()

        self.encoder = encoder(3, 512)
        self.generator = Generator(code_dim, output_size)

        # layers = [PixelNorm()]
        # for i in range(n_mlp):
        #     layers.append(EqualLinear(code_dim, code_dim))
        #     layers.append(nn.LeakyReLU(0.2))

        n_mlp = 3
        code_dim = 512
        layers = [PixelNorm()]
        layers.append(EqualLinear(512 * (input_size // 4) * (input_size // 4), code_dim))
        layers.append(nn.LeakyReLU(0.2))
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)

    def forward(
        self,
        input,
        noise=None,
        noise0=None,
        mean_style=None,
        style_weight=0,
        mixing_range=(-1, -1),
    ):
        step = len(self.generator.progression) - 1
        # if type(z) not in (list, tuple):
        #     z = [z]
        #
        # batch = z[0].shape[0]
        batch = len(input)
        if noise is None:
            noise = []

            for i in range(0, step + 1):
                size = 4 * 2 ** i
                # noise.append(torch.zeros(batch, 1, size, size).cuda())
                noise.append(torch.randn(batch, 1, size, size).cuda())

        m, v = self.encoder(input)
        return m, v, self.generator(self.style, m, v, noise, mixing_range=mixing_range)

    def mean_style(self, input):
        style = self.style(input).mean(0, keepdim=True)

        return style


class Discriminator(nn.Module):
    def __init__(self, initial_res: int = 128, fused=True):
        super().__init__()

        convs = [
            ConvBlock(64, 128, 3, 1, downsample=True, fused=fused),  # 64
            ConvBlock(128, 256, 3, 1, downsample=True, fused=fused),  # 32
            ConvBlock(256, 512, 3, 1, downsample=True),  # 16
        ]
        if initial_res  >= 128:
            convs = convs + [
                ConvBlock(512, 512, 3, 1, downsample=True),  # 8
                ConvBlock(512, 512, 3, 1, downsample=True),  # 4
            ]

        if initial_res >= 256:
            convs = convs + [ConvBlock(512, 512, 3, 1, downsample=True),  ]

        self.progression = nn.ModuleList(
            convs +
            [
                ConvBlock(512, 512, 3, 1, 4, 0, norm=False),
            ]
        )

        self.from_rgb = EqualConv2d(3, 64, 1)
        self.n_layer = len(self.progression)
        self.linear = nn.Sequential(
            EqualLinear(512, 1)
        )

    def forward(self, input):
        step = len(self.progression) - 1
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb(input)

            # if i == 0:
            #     out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
            #     mean_std = out_std.mean()
            #     mean_std = mean_std.expand(out.size(0), 1, 4, 4)
            #     out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

        out = out.squeeze(2).squeeze(2)
        # print(input.size(), out.size(), step)
        out = self.linear(out)

        return out
