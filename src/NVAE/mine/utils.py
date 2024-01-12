import torch


def kl_balancer(kl_unbalanced_terms: torch.Tensor, beta: float = 1.0, balance: bool = False, alpha: torch.Tensor = None):

    if balance and beta < 1.0:

        # done only during training warmup phase

        device = kl_unbalanced_terms.device
        alpha = alpha.to(device)  # terms depending on groups

        kl_terms = torch.mean(kl_unbalanced_terms, dim=0)  # mean on batch
        kl_coefficients = torch.mean(torch.abs(kl_unbalanced_terms), dim=0, keepdim=True)  # proportional to kl_terms

        # set coefficients as summing to 1
        kl_coefficients = kl_coefficients / alpha  # divide by spatial resolution (alpha)
        kl_coefficients = kl_coefficients / torch.sum(kl_coefficients, dim=1, keepdim=True)  # normalize -> sum to 1
        kl_coefficients = kl_coefficients * kl_terms.shape[0]  # sum to num_groups
        total_kl = torch.sum(kl_terms * kl_coefficients.detach(), dim=1)

        # for reporting
        kl_gammas = kl_coefficients.squeeze(0)

    else:

        # after warmup and validation
        total_kl = torch.sum(kl_unbalanced_terms, dim=1)  # sum of each component (not balanced)
        kl_terms = torch.mean(kl_unbalanced_terms, dim=0)  # mean on batch
        kl_gammas = torch.ones_like(kl_terms)

    return beta * total_kl, kl_gammas, kl_terms
