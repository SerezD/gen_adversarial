# code adapted from https://github.com/mgp123/nvae/blob/main/model/distributions.py
import torch
from einops import rearrange, repeat
from torch.nn import functional as F
import numpy as np


def one_hot(indices, depth, dim):
    indices = indices.unsqueeze(dim)
    size = list(indices.size())
    size[dim] = depth
    y_onehot = torch.zeros(size).cuda()
    y_onehot.zero_()
    y_onehot.scatter_(dim, indices, 1)

    return y_onehot


@torch.jit.script
def soft_clamp(x: torch.Tensor, n: float = 5.):
    """
    check https://github.com/NVlabs/NVAE/issues/27

    Clamp values in a differentiable and soft way between [-n; n]: n. * torch.tanh(x / n.)
    This helps a little in maintaining small KL terms.

    Note: why in the source code they used 5. it is not explained.
    """
    return x.div(n).tanh_().mul(n)


# pytorch example for Logistic Distribution --> https://pytorch.org/docs/stable/distributions.html
# why the logistic distribution does not exist in pytorch? --> https://github.com/pytorch/pytorch/issues/7857
# using logistic normal as base, then transform it when mean, scale parameters are picked.
base_distribution = torch.distributions.Uniform(1e-5, 1 - 1e-5)  # avoiding corner values
transforms = [torch.distributions.SigmoidTransform().inv]
base_logistic = torch.distributions.TransformedDistribution(base_distribution, transforms)


class DiscMixLogistic:

    def __init__(self, params: torch.Tensor, num_bits: int = 8):
        """
        Defines 3 params for N Logistics: 1. which to take, 2. mu, 3. log_scale
        Args:
            params: B, N_MIX * 3, C = 3, H, W
            num_bits: final image range goes from [0 to 2. ** num_bits -1)
        """

        # hyperparams
        self.b, N, self.img_channels, self.h, self.w = params.shape
        self.num_mixtures = N // 3
        self.max_val = 2. ** num_bits - 1

        # get params for pi (prob of selecting logistic i), means, log_scales
        p, m, s = torch.chunk(rearrange(params, 'b n c h w -> b n (c h w)'), chunks=3, dim=1)

        # each has shape: B, IMG_C, M, H, W
        self.means_logits = soft_clamp(m)
        self.log_scales_logits = torch.clamp(s, min=-7.0)
        self.logistic_logits = torch.tanh(p)  # [-1, 1.] range

    def log_prob(self, samples: torch.Tensor):
        """
        Args:
            samples: ground truth images in 0_1 range.
        Returns:

        """
        assert torch.max(samples) <= 1.0 and torch.min(samples) >= 0.0

        b, _, h, w = samples.shape

        # convert samples to be in [-1, 1]
        samples = 2 * samples - 1.0

        # expand with num mixtures
        samples = repeat(samples, 'b c h w -> b c M h w', M=self.num_mixtures)

        # center samples given means of each Logistic
        centered = samples - self.means

        # compute CDF
        neg_scale = torch.exp(- self.log_scales)

        plus_in = neg_scale * (centered + 1. / self.max_val)
        cdf_plus = torch.sigmoid(plus_in)

        min_in = neg_scale * (centered - 1. / self.max_val)
        cdf_min = torch.sigmoid(min_in)

        cdf_delta = cdf_plus - cdf_min

        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_min = - F.softplus(min_in)
        mid_in = inverted_scale * centered
        log_pdf_mid = mid_in - self.log_scales - 2. * F.softplus(mid_in)

        log_prob_mid_safe = torch.where(cdf_delta > 1e-5,
                                        torch.log(torch.clamp(cdf_delta, min=1e-10)),
                                        log_pdf_mid - np.log(self.max_val / 2))
        # the original implementation uses samples > 0.999, this ignores the largest possible pixel value (255)
        # which is mapped to 0.9922
        log_probs = torch.where(samples < -0.999, log_cdf_plus, torch.where(samples > 0.99, log_one_minus_cdf_min,
                                                                            log_prob_mid_safe))   # B, 3, M, H, W

        log_probs = torch.sum(log_probs, 1) + F.log_softmax(self.logit_probs, dim=1)  # B, M, H, W
        return torch.logsumexp(log_probs, dim=1)                                      # B, H, W

    def sample(self):

        # B, (C H W) of indices in range 0 __ N_MIXTURES
        logistic_selection_mask = torch.distributions.Categorical(logits=self.logistic_logits.permute(0, 2, 1)).sample()

        # select parameters (B, (C H W))
        selected_mu = torch.gather(self.means_logits, 1, logistic_selection_mask.unsqueeze(1)).squeeze(1)
        selected_scale = torch.gather(self.log_scales_logits, 1, logistic_selection_mask.unsqueeze(1)).squeeze(1)

        # sample and transform
        x = selected_mu + torch.exp(selected_scale) * base_logistic.sample(sample_shape=selected_mu.shape)

        # move to 0-1 range
        x = (x + 1) / 2

        return rearrange(x, 'b (c h w) -> b c h w', c=self.img_channels, h=self.h, w=self.w)

    def mean(self):

        # just scale means by their corresponding probs
        probs = torch.softmax(self.log_scales_logits, dim=1)
        res = self.means_logits * probs
        res = torch.sum(res, dim=1)

        # normalize 0__1 range
        res = (res + 1) / 2

        return rearrange(res, 'b (c h w) -> b c h w', c=self.img_channels, h=self.h, w=self.w)


if __name__ == '__main__':

    gt = torch.rand((4, 3, 32, 32))
    num_bits = 8
    logits = torch.rand((4, 30, 3, 32, 32))

    disc_mixture = DiscMixLogistic(logits, num_bits)

    # training.
    # recon = disc_mixture.log_prob(gt)
    # loss = - torch.sum(recon, dim=[1, 2])  # summation over RGB is done.

    # sampling.
    # img = disc_mixture.sample()
    # img2 = disc_mixture.mean()
