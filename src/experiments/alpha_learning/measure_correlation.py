import torch
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def main():
    """
    Given a set of alpha values and their accuracy.
    1. Compute Spearman's correlation coefficient for each alpha list, to measure monotonicity
    2. Plot monotonicity vs accuracies
    3. Measure the linear correlation between the two
    """

    alphas = np.load('./results/E4E_StyleGAN_resnet-50/alphas.npy')
    alphas = torch.tensor(alphas, device=device)

    b, n = alphas.shape

    x_points = torch.arange(n, device=device).repeat(b, 1)

    rank_x = torch.argsort(x_points, dim=1)  # naively equal to x_points
    rank_y = torch.argsort(alphas, dim=1)

    d_squared = torch.pow(rank_x - rank_y, 2)

    rho = (1 - (6 * d_squared.sum(dim=1)) / (n * (n**2 - 1))).cpu().numpy()
    accuracies = np.load('./results/E4E_StyleGAN_resnet-50/accuracies.npy')[:, 0]

    # compute Pearson linear correlation
    # Compute the covariance of x and y
    covariance = np.mean((rho - rho.mean()) * (accuracies - accuracies.mean()))

    # Compute the standard deviations of x and y
    x_std = np.std(rho)
    y_std = np.std(accuracies)

    # Compute the Pearson correlation coefficient
    pearson_correlation = covariance / (x_std * y_std)

    plt.scatter(rho, accuracies)
    plt.suptitle("Spearman's Correlation Coefficient vs Measured Accuracy\nfor different Alpha combinations\n"
                 f"Measured Pearson Linear Correlation: {pearson_correlation:.4f}")
    plt.xlabel(r'$\rho$')
    plt.ylabel('accuracy')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    main()