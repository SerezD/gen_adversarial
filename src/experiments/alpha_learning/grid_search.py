import argparse
import os

import numpy as np
import torch

from src.experiments.alpha_learning.common_utils import AlphaEvaluator


def parse_args():

    parser = argparse.ArgumentParser('Load an HL purification model and compute some alphas')

    parser.add_argument('--adv_images_path', type=str, required=True,
                        help='Precomputed adversaries to use for evaluation')

    parser.add_argument('--n_steps', type=int, required=True)

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
    args.results_folder = f'{args.results_folder}/{args.autoencoder_name}_{args.classifier_type}/grid_search/'

    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    return args


@torch.no_grad()
def main(args):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    evaluator = AlphaEvaluator(args, device)
    n_alphas = len(evaluator.defense_model.model.interpolation_alphas)

    all_alphas = torch.empty((0, n_alphas), device='cpu')
    all_accuracies = torch.empty((0, 1), device='cpu')

    # Evaluate random alphas
    for s in range(args.n_steps):

        print(f'[INFO] step: {s}')

        # get alpha set
        alphas = torch.zeros((n_alphas,), device=device).uniform_(0, 1)

        # evaluate
        accuracy = evaluator.objective_function(alphas)

        # append and continue
        all_alphas = torch.cat((all_alphas, alphas.cpu().unsqueeze(0)), dim=0)
        all_accuracies = torch.cat((all_accuracies, torch.tensor([[accuracy]])), dim=0)

    # save
    np.save(f'{args.results_folder}/alphas.npy', all_alphas.numpy())
    np.save(f'{args.results_folder}/accuracies.npy', all_accuracies.numpy())


if __name__ == '__main__':

    main(parse_args())
