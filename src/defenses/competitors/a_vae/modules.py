import torch
from torch import nn
from torch.nn import functional
from torch.autograd import Function
from math import sqrt


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


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
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
        out = self.adain1(out, style)

        out = self.conv2(out)
        out = self.noise2(out, noise)
        out = self.lrelu2(out)
        out = self.adain2(out, style)

        return out


class EncodeConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        norm=True,
    ):
        super().__init__()
        self.norm = norm
        self.conv1 = EqualConv2d(
            in_channel, out_channel, kernel_size, stride=1, padding=padding
        )
        self.lrelu1 = nn.LeakyReLU(0.2)
        if norm:
            self.norm1 = nn.InstanceNorm2d(out_channel)

        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, stride=2, padding=padding)
        self.lrelu2 = nn.LeakyReLU(0.2)
        if norm:
            self.norm2 = nn.InstanceNorm2d(out_channel)

    def forward(self, input):
        out = self.conv1(input)
        if self.norm:
            self.norm1(out)
        out = self.lrelu1(out)

        out = self.conv2(out)
        if self.norm:
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
        norm=True,
    ):
        super().__init__()
        self.norm = norm
        self.conv1 = EqualConv2d(
            in_channel, out_channel, kernel_size, stride=1, padding=padding
        )
        self.noise1 = equal_lr(NoiseInjection(out_channel))
        self.lrelu1 = nn.LeakyReLU(0.2)
        if norm:
            self.norm1 = nn.InstanceNorm2d(out_channel)
        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, stride=2, padding=padding)
        self.noise2 = equal_lr(NoiseInjection(out_channel))
        self.lrelu2 = nn.LeakyReLU(0.2)
        if norm:
            self.norm2 = nn.InstanceNorm2d(out_channel)

    def forward(self, input, noise):
        out = self.conv1(input)
        out = self.noise1(out, noise[0])
        if self.norm:
            self.norm1(out)
        out = self.lrelu1(out)
        out = self.conv2(out)
        out = self.noise2(out, noise[1])
        if self.norm:
            self.norm2(out)
        out = self.lrelu2(out)

        return out
