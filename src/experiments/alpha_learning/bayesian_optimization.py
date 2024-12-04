import argparse
import numpy as np
import os
import torch

from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf

from src.experiments.alpha_learning.common_utils import AlphaEvaluator, get_linear_alphas, get_cosine_alphas


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser('Load an MLVGM purification model and learn best alphas')

    parser.add_argument('--adv_images_path', type=str, required=True,
                        help='Precomputed adversaries to use for evaluation')

    parser.add_argument('--n_optimization_steps', type=int, required=True)

    parser.add_argument('--classifier_path', type=str, required=True,
                        help='path to the pre-trained classifier to be attacked')

    parser.add_argument('--classifier_type', type=str, choices=['resnet-50', 'vgg-11', 'resnext-50'],
                        help='type of classifier')

    parser.add_argument('--autoencoder_path', type=str, required=True,
                        help='path to the pre-trained autoencoder acting as a defense')

    parser.add_argument('--autoencoder_name', type=str, required=True,
                        help='used to determine results folder')

    parser.add_argument('--results_folder', type=str, required=True,
                        help='folder to save .numpy files with results')

    args = parser.parse_args()

    # create folder
    args.results_folder = f'{args.results_folder}/{args.autoencoder_name}_{args.classifier_type}/bayesian_optimization/'

    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    return args


def main(args: argparse.Namespace):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    evaluator = AlphaEvaluator(args, device)
    n_alphas = len(evaluator.defense_model.model.interpolation_alphas)

    # Define the search space for the hyperparameters
    hp_bounds = torch.tensor([[0.0] * n_alphas, [1.0] * n_alphas], device=device)

    # Initial alphas
    print('[INFO] Initializing...')
    train_x = torch.tensor([
        get_cosine_alphas(n_alphas),
        get_linear_alphas(n_alphas),
        [0.5 for _ in range(n_alphas)],
        [1 - i for i in get_linear_alphas(n_alphas)],
        [1 - i for i in get_cosine_alphas(n_alphas)]
    ], device=device)

    train_y = []
    for x in train_x:
        y = evaluator.objective_function(x)
        train_y.append([1 - y])
        print(f'alphas: {x.cpu().tolist()}; accuracy: {y}')

    train_y = torch.tensor(train_y, device=device)

    # Initialize the Gaussian Process model
    gp = SingleTaskGP(train_x.to(torch.double), train_y.to(torch.double))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # Define the acquisition function (Expected Improvement)
    best_f = train_y.min().item()
    acq_func = ExpectedImprovement(gp, best_f=best_f, maximize=False)

    # Bayesian Optimization loop
    for s in range(args.n_optimization_steps):

        print(f'[INFO] step: {s}')

        # Optimize the acquisition function
        candidate, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=hp_bounds,
            q=1,
            num_restarts=8,
            raw_samples=32,
        )

        # Evaluate the objective function on the new candidate
        new_x = candidate.detach()
        new_y = torch.tensor([[1. - evaluator.objective_function(new_x[0])]], device=device)

        # Update the training data
        train_x = torch.cat([train_x, new_x], dim=0)
        train_y = torch.cat([train_y, new_y], dim=0)

        # Refit the GP model
        gp = SingleTaskGP(train_x.to(torch.double), train_y.to(torch.double))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # Update the acquisition function with new best observed value
        best_f = train_y.min().item()
        acq_func = ExpectedImprovement(gp, best_f=best_f, maximize=False)

    # The best set of hyperparameters found
    best_hyperparameters = train_x[train_y.argmin()]
    print(f'best alphas: {best_hyperparameters.cpu().tolist()} - accuracy: {1 - train_y.min().item()}')

    # save
    np.save(f'{args.results_folder}/alphas.npy', train_x.cpu().numpy())
    np.save(f'{args.results_folder}/accuracies.npy', 1 - train_y.cpu().numpy())


if __name__ == '__main__':

    main(parse_args())
