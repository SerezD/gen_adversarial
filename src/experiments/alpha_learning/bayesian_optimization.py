import argparse
import os

import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import ImageLabelDataset
from src.defenses.models import CelebAResnetModel, CelebAStyleGanDefenseModel
from src.experiments.alpha_learning.utils import get_linear_alphas, get_cosine_alphas


def parse_args():

    parser = argparse.ArgumentParser('Load an HL purification model and learn best alphas')

    parser.add_argument('--adv_images_path', type=str, required=True,
                        help='Precomputed adversaries to use for evaluation')

    parser.add_argument('--batch_size', type=int, required=True)

    parser.add_argument('--n_optimization_steps', type=int, required=True)

    parser.add_argument('--classifier_path', type=str, required=True,
                        help='path to the pre-trained classifier to be attacked')

    parser.add_argument('--classifier_type', type=str, choices=['resnet-50'],
                        help='type of classifier')

    parser.add_argument('--autoencoder_path', type=str, required=True,
                        help='path to the pre-trained autoencoder acting as a defense')

    parser.add_argument('--autoencoder_name', type=str, required=True,
                        help='used to determine results folder')

    parser.add_argument('--results_folder', type=str, required=True,
                        help='folder to save .numpy files with results')

    args = parser.parse_args()

    # create folder
    args.results_folder = f'{args.results_folder}/{args.autoencoder_name}_{args.classifier_type}/'

    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    return args


class AlphaEvaluator:
    def __init__(self, args, device):

        self.device = device

        # load model
        if args.classifier_type == 'resnet-50':
            args.image_size = 256
            base_classifier = CelebAResnetModel(args.classifier_path, device)
            alphas = [0. for _ in range(18)]
            self.defense_model = CelebAStyleGanDefenseModel(base_classifier, args.autoencoder_path, alphas,
                                                            initial_noise_eps=0.0, apply_gaussian_blur=False,
                                                            device=device)
        else:
            raise ValueError(f'Unknown classifier type: {args.classifier_type}')

        # get dataloader
        dataset = ImageLabelDataset(folder=args.adv_images_path, image_size=args.image_size)
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    @torch.no_grad()
    def objective_function(self, alphas):
        """
        One Epoch of Defense Model with given alphas on the pre-computed adversarial set
        """
        self.defense_model.interpolation_alphas = alphas.cpu().tolist()

        accuracy = None

        for idx, batch in enumerate(tqdm(self.dataloader)):

            x, y = batch[0].to(self.device), batch[1].to(self.device)

            # average result
            n_times_preds = [torch.argmax(self.defense_model(x), dim=-1) for _ in range(6)]
            preds = torch.mode(torch.stack(n_times_preds, dim=0), dim=0).values

            accuracy = torch.eq(preds, y) if accuracy is None else torch.cat((accuracy, preds), dim=0)

            # if idx > 4:
            #     break

        accuracy = torch.mean(accuracy.to(torch.float32)).item()
        return 1 - accuracy

def main(args):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    evaluator = AlphaEvaluator(args, device)
    n_alphas = len(evaluator.defense_model.interpolation_alphas)

    # Define the search space for the hyperparameters
    hp_bounds = torch.tensor([[0.0] * n_alphas, [1.0] * n_alphas], device=device)

    # Initial alphas
    print('[INFO] Initializing...')
    train_x = torch.tensor([
        get_cosine_alphas(n_alphas),
        get_linear_alphas(n_alphas),
        [0.5 for i in range(n_alphas)],
        [1 - i for i in get_linear_alphas(n_alphas)],
        [1 - i for i in get_cosine_alphas(n_alphas)]
    ], device=device)

    train_y = torch.tensor(
        [[evaluator.objective_function(x)] for x in train_x], device=device)

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
        new_y = torch.tensor([[evaluator.objective_function(new_x)]], device=device)

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
