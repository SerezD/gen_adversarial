import argparse

import numpy as np
import torch
from matplotlib import pyplot as plt
import os
import pickle


def parse_args():

    parser = argparse.ArgumentParser('Plot KL divergence heatmaps')

    parser.add_argument('--results_folder', type=str,
                        required=True, help='folder with .pickle files of pre-computed train val codes and subfolders'
                                            'of attack codes.')

    parser.add_argument('--cnn_type', type=str, choices=['resnet-32', 'vgg-16', 'resnet-50'],
                        help='name of CNN that has been evaluated')

    parser.add_argument('--l2_bound', type=float,
                        help='plot results for this bound.')

    args = parser.parse_args()
    if args.results_folder[-1] == '/':
        args.results_folder = args.results_folder[:-1]

    args.autoencoder_name = args.results_folder.split('/')[-1]

    return args


def main(args: argparse.Namespace):

    # load base 'train' samples
    with open(f'{args.results_folder}/train_codes.pickle', 'rb') as f:
        train_codes = pickle.load(f)

    # load clean 'valid' samples
    attack_codes = []
    with open(f'{args.results_folder}/valid_codes.pickle', 'rb') as f:
        valid_codes = pickle.load(f)
        attack_codes.append(valid_codes)

    # load samples for each attack
    attacks = ('BruteForce', 'DeepFool', 'C&W')
    subfolder = f'{args.results_folder}/{args.cnn_type}/'

    attack_files = [f'{subfolder}{a}_codes_l2={args.l2_bound}.pickle' for a in attacks]
    for a_f in attack_files:
        with open(a_f, 'rb') as f:
            noisy_codes = pickle.load(f)
            attack_codes.append(noisy_codes)

    # compute KL divergences for random noise
    latents = list(train_codes.keys())[::-1]
    kl_matrix = torch.zeros((len(latents), len(attack_codes)))

    for idx_latent in latents:

        # get P distribution from train set.
        train_dist = torch.tensor(train_codes[idx_latent])
        P_distribution = torch.distributions.MultivariateNormal(train_dist.mean(dim=0),
                                                                covariance_matrix=torch.diag(train_dist.var(dim=0)))

        for col, noisy_codes in enumerate(attack_codes):

            # get Q (test distribution)
            noisy_dist = torch.tensor(noisy_codes[idx_latent])
            Q_distribution = torch.distributions.MultivariateNormal(noisy_dist.mean(dim=0),
                                                                    covariance_matrix=torch.diag(noisy_dist.var(dim=0)))

            kl_divergence = torch.distributions.kl_divergence(P_distribution, Q_distribution).item()
            kl_matrix[idx_latent, col] = kl_divergence

    # plot matrix for random noise
    fig = plt.figure(figsize=(6, 0.3 * len(latents) + 2))
    plt.imshow(kl_matrix.cpu().numpy(), cmap='YlOrRd', interpolation='nearest', aspect='auto')

    plt.xticks(range(len(attacks) + 1), ('Valid',) + attacks)
    plt.yticks(range(len(latents)), [str(i) for i in range(len(latents))])
    plt.xlabel(f'Attack (L2 Bound = {args.l2_bound})')
    plt.ylabel('Latent Index')

    plt.colorbar(label='KL divergence')

    # Add value annotations
    for i in range(kl_matrix.shape[0]):
        for j in range(kl_matrix.shape[1]):
            plt.text(j, i, f'{kl_matrix[i, j]:.2f}', ha='center', va='center', color='black')

    plt.suptitle(f'KL divergence per latent of {args.autoencoder_name}\nafter attacks on {args.cnn_type}')
    plt.tight_layout()
    plt.show()
    plt.close(fig)


if __name__ == '__main__':

    arguments = parse_args()

    main(arguments)
