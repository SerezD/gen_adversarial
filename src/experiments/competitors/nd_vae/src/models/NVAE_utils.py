
''' Some utils taken from the NVAE code:
https://github.com/NVlabs/NVAE/blob/9fc1a288fb831c87d93a4e2663bc30ccf9225b29/utils.py#L161'''


import torch
import numpy as np


import torch.nn.functional as F
from einops import pack, rearrange, repeat


#from tensorboardX import SummaryWriter


def kl_per_group(kl_all):
    kl_vals = torch.mean(kl_all, dim=0)
    kl_coeff_i = torch.abs(kl_all)
    kl_coeff_i = torch.mean(kl_coeff_i, dim=0, keepdim=True) + 0.01

    return kl_coeff_i, kl_vals



def kl_balancer(kl_all, kl_coeff=1.0, kl_balance=False, alpha_i=None):
    if kl_balance and kl_coeff < 1.0:
        #alpha_i = alpha_i.unsqueeze(0)
        alpha_i = alpha_i[1:]
        alpha_i = alpha_i.unsqueeze(0)

        kl_all = torch.stack(kl_all, dim=1)
        kl_coeff_i, kl_vals = kl_per_group(kl_all)
        total_kl = torch.sum(kl_coeff_i)

        kl_coeff_i = kl_coeff_i[0]
        kl_coeff_i = kl_coeff_i / alpha_i * total_kl
        kl_coeff_i = kl_coeff_i / torch.mean(kl_coeff_i, dim=1, keepdim=True)
        kl = torch.sum(kl_all * kl_coeff_i.detach(), dim=1)

        # for reporting
        kl_coeffs = kl_coeff_i.squeeze(0)
    else:
        kl_all = torch.stack(kl_all, dim=1)
        kl_vals = torch.mean(kl_all, dim=0)
        kl = torch.sum(kl_all, dim=1)
        kl_coeffs = torch.ones(size=(len(kl_vals),))

    return kl_coeff * kl, kl_coeffs, kl_vals


def kl_coeff(step, total_step, constant_step, min_kl_coeff):
    return max(min((step - constant_step) / total_step, 1.0), min_kl_coeff)

def decode_output(recon_x):
    decoder = NormalDecoder(recon_x)

    #recon = decoder.log_prob(x)
    recon = decoder.sample()
    return recon

def reconstruction_loss(recon_x, x, crop=False):

    recon_x = decode_output(recon_x)
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    return BCE


def gumbel_sampling(logits: torch.Tensor, temperature: float = 1.0):
    """
    better that categorical sampling
    """

    one_hot_probs = torch.zeros_like(logits)

    gumbel_noise = torch.zeros_like(logits).uniform_(1e-5, 1. - 1e-5)
    gumbel_noise = -torch.log(-torch.log(gumbel_noise))
    logits = torch.argmax(logits / temperature + gumbel_noise, 1, keepdim=True)

    return one_hot_probs.scatter_(1, logits, 1)


class DiscMixLogistic:
    """
    code adapted from https://github.com/mgp123/nvae/blob/main/model/distributions.py
    and from https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
    """

    def __init__(self, params: torch.Tensor, img_channels: int = 3, num_bits: int = 8):
        """
        Defines 3 params for N Logistics: 1. which to take, 2. mu, 3. log_scale
        Additionally dimension is the final logits probabilities
        Args:
            params: B, N_MIX + N_MIX * 3 * C, H, W
            num_bits: final image range goes from [0 to 2. ** num_bits -1)
        """

        if img_channels != 3:
            raise NotImplementedError("actually works only with 3D images")

        # hyperparams
        self.img_channels = img_channels
        self.b, x, self.h, self.w = params.shape
        self.num_mixtures = x // (1 + img_channels * 3)
        self.max_val = 2. ** num_bits - 1

        self.logits = rearrange(params[:, :self.num_mixtures], 'b n h w -> b n (h w)')

        # get params for logistics means, log_scales and coefficients
        params = rearrange(params[:, self.num_mixtures:], 'b (n c) h w -> b n c (h w)', n=self.num_mixtures)
        m, s, k = torch.chunk(params, chunks=3, dim=2)

        self.means_logits = m
        self.log_scales_logits = torch.clamp(s, min=-7.0)
        self.logistic_coefficients = torch.tanh(k)  # [-1, 1.] range

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
        samples = repeat(samples, 'b c h w -> b n c (h w)', n=self.num_mixtures)

        # adjust means based on preceding channels (check section 2.2 of PixelCNN++ paper)
        # https://openreview.net/pdf?id=BJrFC6ceg
        r_mean = self.means_logits[:, :, 0, :].unsqueeze(2)  # B, N, RED_CH, (H, W)

        # B, N, GREEN_CH, (H, W)
        g_mean = self.means_logits[:, :, 1, :].unsqueeze(2)
        g_mean = g_mean + (self.logistic_coefficients[:, :, 0, :] * samples[:, :, 0, :]).unsqueeze(2)

        # B, N, BLUE_CH, (H, W)
        b_mean = self.means_logits[:, :, 2, :].unsqueeze(2)
        b_mean = b_mean + (self.logistic_coefficients[:, :, 1, :] * samples[:, :, 0, :]).unsqueeze(2)
        b_mean = b_mean + (self.logistic_coefficients[:, :, 2, :] * samples[:, :, 1, :]).unsqueeze(2)

        adjusted_means, _ = pack([r_mean, g_mean, b_mean], 'b n * d')  # B N C (H W)

        # compute CDF in the neighborhood of each sample
        centered = samples - adjusted_means
        neg_scale = torch.exp(- self.log_scales_logits)

        plus_in = neg_scale * (centered + 1. / self.max_val)
        cdf_plus = torch.sigmoid(plus_in)

        min_in = neg_scale * (centered - 1. / self.max_val)
        cdf_min = torch.sigmoid(min_in)

        # log probability for edge case of 0
        log_cdf_plus = plus_in - torch.nn.functional.softplus(plus_in)

        # log probability for edge case of 255
        log_one_minus_cdf_min = - torch.nn.functional.softplus(min_in)

        # probability for all other cases
        cdf_delta = cdf_plus - cdf_min

        # probability for very low values (avoid nan in following computations)
        safe_log_probs = neg_scale * centered
        safe_log_probs = safe_log_probs - self.log_scales_logits - 2. * torch.nn.functional.softplus(safe_log_probs)
        safe_log_probs = safe_log_probs - np.log(self.max_val / 2)

        # select the right output: left edge case, right edge case, normal case (filtered for low values)
        safe_log_probs = torch.where(cdf_delta > 1e-5,
                                     torch.log(torch.clamp(cdf_delta, min=1e-10)),
                                     safe_log_probs)

        log_probs = torch.where(samples < -0.999,
                                log_cdf_plus,
                                torch.where(samples > 0.99,
                                            log_one_minus_cdf_min,
                                            safe_log_probs)
                                )

        # log_probs is B N C (H W) --> sum over channels
        # logits is B N (H W)
        log_probs = torch.sum(log_probs, dim=2) + torch.nn.functional.log_softmax(self.logits, dim=1)
        return torch.logsumexp(log_probs, dim=1)

    def sample(self):

        # Gumbel Sampling of shape B, (H W) of indices in range 0 __ N_MIXTURES
        logistic_selection_mask = gumbel_sampling(self.logits).unsqueeze(2)

        # select parameters and sum on NUM_MIXTURES --> B, C, (H W)
        selected_mu = torch.sum(self.means_logits * logistic_selection_mask, dim=1)
        selected_scale = torch.sum(self.log_scales_logits * logistic_selection_mask, dim=1)
        selected_k = torch.sum(self.logistic_coefficients * logistic_selection_mask, dim=1)

        # sample from logistic with default parameters and then scale
        # pytorch example for Logistic Distribution --> https://pytorch.org/docs/stable/distributions.html
        # why the logistic distribution does not exist in pytorch? --> https://github.com/pytorch/pytorch/issues/7857
        base_sample = torch.zeros_like(selected_mu).uniform_(1e-5, 1. - 1e-5)  # avoiding corner values
        base_sample = torch.log(base_sample) - torch.log(1 - base_sample)

        x = selected_mu + torch.exp(selected_scale) * base_sample  # B, C, (H W)

        # adjust given previous channels (clamp in -1, 1. range)
        r_x = torch.clamp(x[:, 0, :], -1., 1.)  # B, (H, W)

        g_x = x[:, 1, :] + selected_k[:, 0, :] * r_x
        g_x = torch.clamp(g_x, -1., 1.)  # B, (H, W)

        b_x = x[:, 2, :] + selected_k[:, 1, :] * r_x + selected_k[:, 2, :] * g_x
        b_x = torch.clamp(b_x, -1., 1.)  # B, (H, W)

        # move to 0-1 range --> B 3 (H W)
        x = (pack([r_x.unsqueeze(1), g_x.unsqueeze(1), b_x.unsqueeze(1)], 'b * d')[0] + 1) / 2

        return rearrange(x, 'b c (h w) -> b c h w', h=self.h, w=self.w)

    def mean(self):

        # just scale means by their corresponding probs
        probs = torch.softmax(self.logits, dim=1).unsqueeze(2)

        # select logistic parameters
        selected_mu = torch.sum(self.means_logits * probs, dim=1)  # B, 3, (H, W)
        selected_k = torch.sum(self.logistic_coefficients * probs, dim=1)  # B, 3, (H, W)

        # don't sample, use mean
        x = selected_mu

        # adjust given previous channels (clamp in -1, 1. range)
        r_x = torch.clamp(x[:, 0, :], -1., 1.)  # B, (H, W)

        g_x = x[:, 1, :] + selected_k[:, 0, :] * r_x
        g_x = torch.clamp(g_x, -1., 1.)  # B, (H, W)

        b_x = x[:, 2, :] + selected_k[:, 1, :] * r_x + selected_k[:, 2, :] * g_x
        b_x = torch.clamp(b_x, -1., 1.)  # B, (H, W)

        # move to 0-1 range --> B 3 (H W)
        x = (pack([r_x.unsqueeze(1), g_x.unsqueeze(1), b_x.unsqueeze(1)], 'b * d')[0] + 1) / 2

        return rearrange(x, 'b c (h w) -> b c h w', h=self.h, w=self.w)


def kl_balancer_coeff(num_scales, groups_per_scale, fun):

    #NOTE: In NVAE code, this list is originally called groups_per_scale
    groups_list = []
    for i in range(num_scales):
        groups_list.append(groups_per_scale)

    if fun == 'equal':
        coeff = torch.cat([torch.ones(groups_list[num_scales - i - 1]) for i in range(num_scales)], dim=0).cuda()
    elif fun == 'linear':
        coeff = torch.cat([(2 ** i) * torch.ones(groups_list[num_scales - i - 1]) for i in range(num_scales)], dim=0).cuda()
    elif fun == 'sqrt':
        coeff = torch.cat([np.sqrt(2 ** i) * torch.ones(groups_list[num_scales - i - 1]) for i in range(num_scales)], dim=0).cuda()
    elif fun == 'square':
        coeff = torch.cat([np.square(2 ** i) / groups_list[num_scales - i - 1] * torch.ones(groups_list[num_scales - i - 1]) for i in range(num_scales)], dim=0).cuda()
    else:
        raise NotImplementedError
    # convert min to 1.
    coeff /= torch.min(coeff)
    return coeff

@torch.jit.script
def soft_clamp5(x: torch.Tensor):
    return x.div(5.).tanh_().mul(5.)    #  5. * torch.tanh(x / 5.) <--> soft differentiable clamp between [-5, 5]


@torch.jit.script
def sample_normal_jit(mu, sigma):
    eps = mu.mul(0).normal_()
    z = eps.mul_(sigma).add_(mu)
    return z, eps

class Normal:
    def __init__(self, mu, log_sigma, temp=1.):
        self.mu = soft_clamp5(mu)
        log_sigma = soft_clamp5(log_sigma)
        self.sigma = torch.exp(log_sigma) + 1e-2      # we don't need this after soft clamp
        if temp != 1.:
            self.sigma *= temp

    def sample(self):
        return sample_normal_jit(self.mu, self.sigma)

    def sample_given_eps(self, eps):
        return eps * self.sigma + self.mu

    def log_p(self, samples):
        normalized_samples = (samples - self.mu) / self.sigma
        log_p = - 0.5 * normalized_samples * normalized_samples - 0.5 * np.log(2 * np.pi) - torch.log(self.sigma)
        return log_p

    def kl(self, normal_dist):
        term1 = (self.mu - normal_dist.mu) / normal_dist.sigma
        term2 = self.sigma / normal_dist.sigma

        return 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)


# class DiscMixLogistic:
#     def __init__(self, param, num_mix=10, num_bits=8):
#         B, C, H, W = param.size()
#         self.num_mix = num_mix
#         self.logit_probs = param[:, :num_mix, :, :]                                   # B, M, H, W
#         l = param[:, num_mix:, :, :].view(B, 3, 3 * num_mix, H, W)                    # B, 3, 3 * M, H, W
#         self.means = l[:, :, :num_mix, :, :]                                          # B, 3, M, H, W
#         self.log_scales = torch.clamp(l[:, :, num_mix:2 * num_mix, :, :], min=-7.0)   # B, 3, M, H, W
#         self.coeffs = torch.tanh(l[:, :, 2 * num_mix:3 * num_mix, :, :])              # B, 3, M, H, W
#         self.max_val = 2. ** num_bits - 1
#
#     def log_prob(self, samples):
#         assert torch.max(samples) <= 1.0 and torch.min(samples) >= 0.0
#         # convert samples to be in [-1, 1]
#         samples = 2 * samples - 1.0
#
#         B, C, H, W = samples.size()
#         assert C == 3, 'only RGB images are considered.'
#
#         samples = samples.unsqueeze(4)                                                  # B, 3, H , W
#         samples = samples.expand(-1, -1, -1, -1, self.num_mix).permute(0, 1, 4, 2, 3)   # B, 3, M, H, W
#         mean1 = self.means[:, 0, :, :, :]                                               # B, M, H, W
#         mean2 = self.means[:, 1, :, :, :] + \
#                 self.coeffs[:, 0, :, :, :] * samples[:, 0, :, :, :]                     # B, M, H, W
#         mean3 = self.means[:, 2, :, :, :] + \
#                 self.coeffs[:, 1, :, :, :] * samples[:, 0, :, :, :] + \
#                 self.coeffs[:, 2, :, :, :] * samples[:, 1, :, :, :]                     # B, M, H, W
#
#         mean1 = mean1.unsqueeze(1)                          # B, 1, M, H, W
#         mean2 = mean2.unsqueeze(1)                          # B, 1, M, H, W
#         mean3 = mean3.unsqueeze(1)                          # B, 1, M, H, W
#         means = torch.cat([mean1, mean2, mean3], dim=1)     # B, 3, M, H, W
#         centered = samples - means                          # B, 3, M, H, W
#
#         inv_stdv = torch.exp(- self.log_scales)
#         plus_in = inv_stdv * (centered + 1. / self.max_val)
#         cdf_plus = torch.sigmoid(plus_in)
#         min_in = inv_stdv * (centered - 1. / self.max_val)
#         cdf_min = torch.sigmoid(min_in)
#         log_cdf_plus = plus_in - F.softplus(plus_in)
#         log_one_minus_cdf_min = - F.softplus(min_in)
#         cdf_delta = cdf_plus - cdf_min
#         mid_in = inv_stdv * centered
#         log_pdf_mid = mid_in - self.log_scales - 2. * F.softplus(mid_in)
#
#         log_prob_mid_safe = torch.where(cdf_delta > 1e-5,
#                                         torch.log(torch.clamp(cdf_delta, min=1e-10)),
#                                         log_pdf_mid - np.log(self.max_val / 2))
#         # the original implementation uses samples > 0.999, this ignores the largest possible pixel value (255)
#         # which is mapped to 0.9922
#         log_probs = torch.where(samples < -0.999, log_cdf_plus, torch.where(samples > 0.99, log_one_minus_cdf_min,
#                                                                             log_prob_mid_safe))   # B, 3, M, H, W
#
#         log_probs = torch.sum(log_probs, 1) + F.log_softmax(self.logit_probs, dim=1)  # B, M, H, W
#         return torch.logsumexp(log_probs, dim=1)                                      # B, H, W
#
#
#     def sample(self, t=1.):
#         gumbel = -torch.log(- torch.log(torch.Tensor(self.logit_probs.size()).uniform_(1e-5, 1. - 1e-5).cuda()))  # B, M, H, W
#         sel = one_hot(torch.argmax(self.logit_probs / t + gumbel, 1), self.num_mix, dim=1)          # B, M, H, W
#         sel = sel.unsqueeze(1)                                                                 # B, 1, M, H, W
#
#         # select logistic parameters
#         means = torch.sum(self.means * sel, dim=2)                                             # B, 3, H, W
#         log_scales = torch.sum(self.log_scales * sel, dim=2)                                   # B, 3, H, W
#         coeffs = torch.sum(self.coeffs * sel, dim=2)                                           # B, 3, H, W
#
#         # cells from logistic & clip to interval
#         # we don't actually round to the nearest 8bit value when sampling
#         u = torch.Tensor(means.size()).uniform_(1e-5, 1. - 1e-5).cuda()                        # B, 3, H, W
#         x = means + torch.exp(log_scales) / t * (torch.log(u) - torch.log(1. - u))             # B, 3, H, W
#
#         x0 = torch.clamp(x[:, 0, :, :], -1, 1.)                                                # B, H, W
#         x1 = torch.clamp(x[:, 1, :, :] + coeffs[:, 0, :, :] * x0, -1, 1)                       # B, H, W
#         x2 = torch.clamp(x[:, 2, :, :] + coeffs[:, 1, :, :] * x0 + coeffs[:, 2, :, :] * x1, -1, 1)  # B, H, W
#
#         x0 = x0.unsqueeze(1)
#         x1 = x1.unsqueeze(1)
#         x2 = x2.unsqueeze(1)
#
#         x = torch.cat([x0, x1, x2], 1)
#         x = x / 2. + 0.5
#         return x
#
#     def mean(self):
#         sel = torch.softmax(self.logit_probs, dim=1)                                           # B, M, H, W
#         sel = sel.unsqueeze(1)                                                                 # B, 1, M, H, W
#
#         # select logistic parameters
#         means = torch.sum(self.means * sel, dim=2)                                             # B, 3, H, W
#         coeffs = torch.sum(self.coeffs * sel, dim=2)                                           # B, 3, H, W
#
#         # we don't sample from logistic components, because of the linear dependencies, we use mean
#         x = means                                                                              # B, 3, H, W
#         x0 = torch.clamp(x[:, 0, :, :], -1, 1.)                                                # B, H, W
#         x1 = torch.clamp(x[:, 1, :, :] + coeffs[:, 0, :, :] * x0, -1, 1)                       # B, H, W
#         x2 = torch.clamp(x[:, 2, :, :] + coeffs[:, 1, :, :] * x0 + coeffs[:, 2, :, :] * x1, -1, 1)  # B, H, W
#
#         x0 = x0.unsqueeze(1)
#         x1 = x1.unsqueeze(1)
#         x2 = x2.unsqueeze(1)
#
#         x = torch.cat([x0, x1, x2], 1)
#         x = x / 2. + 0.5
#         return x

class NormalDecoder:
    def __init__(self, param, num_bits=8):
        B, C, H, W = param.size()
        self.num_c = C // 2
        mu = param[:, :self.num_c, :, :]  # B, 3, H, W
        log_sigma = param[:, self.num_c:, :, :]  # B, 3, H, W
        self.dist = Normal(mu, log_sigma)

    def log_prob(self, samples):
        assert torch.max(samples) <= 1.0 and torch.min(samples) >= 0.0
        # convert samples to be in [-1, 1]
        samples = 2 * samples - 1.0

        return self.dist.log_p(samples)

    def sample(self, t=1.):
        x, _ = self.dist.sample()
        x = torch.clamp(x, -1, 1.)
        x = x / 2. + 0.5
        return x



