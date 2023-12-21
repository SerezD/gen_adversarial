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

class DiscMixLogistic:

    def __init__(self, params: torch.Tensor, num_bits: int = 8):
        """
        Defines 3 params for N Logistics: 1. which to take, 2. mu, 3. log_scale
        Args:
            params: B, C = 3, N_MIX * 3, H, W
            num_bits: final image range goes from [0 to 2. ** num_bits -1)
        """

        # hyperparams
        b, img_channels, N, h, w = params.shape
        self.num_mixtures = N // 3
        self.max_val = 2. ** num_bits - 1

        # B, M, H, W
        # self.logit_probs = logits[:, :num_mixtures, :, :]

        # get params for pi (prob of selecting logistic i), means, log_scales
        p, m, s = torch.chunk(params, chunks=3, dim=2)

        # each has shape: B, IMG_C, M, H, W
        self.means = soft_clamp(m)
        self.log_scales = torch.clamp(s, min=-7.0)
        self.logistic_probs = torch.tanh(p)  # [-1, 1.] range

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

        # ORIGINAL NVAE CODE ( don't understand what is the motivation for this)
        # CHECK updates of issue: https://github.com/NVlabs/NVAE/issues/46
        # mean1 = self.means[:, 0, :, :, :]                                               # B, M, H, W
        # mean2 = self.means[:, 1, :, :, :] + \
        #         self.logistic_probs[:, 0, :, :, :] * samples[:, 0, :, :, :]                     # B, M, H, W
        # mean3 = self.means[:, 2, :, :, :] + \
        #         self.logistic_probs[:, 1, :, :, :] * samples[:, 0, :, :, :] + \
        #         self.logistic_probs[:, 2, :, :, :] * samples[:, 1, :, :, :]                     # B, M, H, W
        #
        # mean1 = mean1.unsqueeze(1)                          # B, 1, M, H, W
        # mean2 = mean2.unsqueeze(1)                          # B, 1, M, H, W
        # mean3 = mean3.unsqueeze(1)                          # B, 1, M, H, W
        # means = torch.cat([mean1, mean2, mean3], dim=1)     # B, 3, M, H, W
        # centered = samples - means                          # B, 3, M, H, W

        # compute CDF
        inverted_scale = torch.exp(- self.log_scales)

        plus_in = inverted_scale * (centered + 1. / self.max_val)
        cdf_plus = torch.sigmoid(plus_in)

        min_in = inverted_scale * (centered - 1. / self.max_val)
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

    def sample(self, t=1.):
        gumbel = -torch.log(- torch.log(torch.Tensor(self.logit_probs.size()).uniform_(1e-5, 1. - 1e-5).cuda()))  # B, M, H, W
        sel = one_hot(torch.argmax(self.logit_probs / t + gumbel, 1), self.num_mixtures, dim=1)          # B, M, H, W
        sel = sel.unsqueeze(1)                                                                 # B, 1, M, H, W

        # select logistic parameters
        means = torch.sum(self.means * sel, dim=2)                                             # B, 3, H, W
        log_scales = torch.sum(self.log_scales * sel, dim=2)                                   # B, 3, H, W
        coeffs = torch.sum(self.logistic_probs * sel, dim=2)                                           # B, 3, H, W

        # cells from logistic & clip to interval
        # we don't actually round to the nearest 8bit value when sampling
        u = torch.Tensor(means.size()).uniform_(1e-5, 1. - 1e-5).cuda()                        # B, 3, H, W
        x = means + torch.exp(log_scales) / t * (torch.log(u) - torch.log(1. - u))             # B, 3, H, W

        x0 = torch.clamp(x[:, 0, :, :], -1, 1.)                                                # B, H, W
        x1 = torch.clamp(x[:, 1, :, :] + coeffs[:, 0, :, :] * x0, -1, 1)                       # B, H, W
        x2 = torch.clamp(x[:, 2, :, :] + coeffs[:, 1, :, :] * x0 + coeffs[:, 2, :, :] * x1, -1, 1)  # B, H, W

        x0 = x0.unsqueeze(1)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

        x = torch.cat([x0, x1, x2], 1)
        x = x / 2. + 0.5
        return x

    def mean(self):
        sel = torch.softmax(self.logit_probs, dim=1)                                           # B, M, H, W
        sel = sel.unsqueeze(1)                                                                 # B, 1, M, H, W

        # select logistic parameters
        means = torch.sum(self.means * sel, dim=2)                                             # B, 3, H, W
        coeffs = torch.sum(self.logistic_probs * sel, dim=2)                                           # B, 3, H, W

        # we don't sample from logistic components, because of the linear dependencies, we use mean
        x = means                                                                              # B, 3, H, W
        x0 = torch.clamp(x[:, 0, :, :], -1, 1.)                                                # B, H, W
        x1 = torch.clamp(x[:, 1, :, :] + coeffs[:, 0, :, :] * x0, -1, 1)                       # B, H, W
        x2 = torch.clamp(x[:, 2, :, :] + coeffs[:, 1, :, :] * x0 + coeffs[:, 2, :, :] * x1, -1, 1)  # B, H, W

        x0 = x0.unsqueeze(1)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

        x = torch.cat([x0, x1, x2], 1)
        x = x / 2. + 0.5
        return x


if __name__ == '__main__':

    gt = torch.rand((4, 3, 32, 32))
    num_mixtures = 10
    num_bits = 8
    logits = torch.rand((4, 10 * num_mixtures, 32, 32))

    disc_mixture = DiscMixLogistic(logits, num_mixtures, num_bits)

    # training.
    recon = disc_mixture.log_prob(gt)
    loss = - torch.sum(recon, dim=[1, 2])  # summation over RGB is done.

    # sampling.
    img = disc_mixture.sample()
    img2 = disc_mixture.mean()
