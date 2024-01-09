import torch


def kl_per_group(kl_all: torch.Tensor):

    kl_vals = torch.mean(kl_all, dim=0)
    kl_coeff_i = torch.mean(torch.abs(kl_all), dim=0, keepdim=True) + 0.01

    return kl_coeff_i, kl_vals


def kl_balancer(kl_all: list, kl_coeff: float = 1.0, kl_balance: bool = False, alpha: torch.Tensor = None):

    device = kl_all[0].device
    if kl_balance and kl_coeff < 1.0:

        # done only during warmup phase
        alpha = alpha.unsqueeze(0).to(device)

        kl_all = torch.stack(kl_all, dim=1)
        kl_coeff_i, kl_vals = kl_per_group(kl_all)
        total_kl = torch.sum(kl_coeff_i)

        kl_coeff_i = kl_coeff_i / alpha * total_kl
        kl_coeff_i = kl_coeff_i / torch.mean(kl_coeff_i, dim=1, keepdim=True)
        kl = torch.sum(kl_all * kl_coeff_i.detach(), dim=1)

        # for reporting
        kl_coeffs = kl_coeff_i.squeeze(0)

    else:

        # during all the rest of training
        kl_all = torch.stack(kl_all, dim=1)
        kl_vals = torch.mean(kl_all, dim=0)
        kl = torch.sum(kl_all, dim=1)
        kl_coeffs = torch.ones(size=(len(kl_vals),))

    return kl_coeff * kl, kl_coeffs, kl_vals