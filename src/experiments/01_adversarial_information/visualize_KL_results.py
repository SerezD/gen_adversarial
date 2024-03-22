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

    args = parser.parse_args()
    if args.results_folder[-1] == '/':
        args.results_folder = args.results_folder[:-1]

    args.autoencoder_name = args.results_folder.split('/')[-1]

    return args


def main(args: argparse.Namespace):

    # load clean samples
    with open(f'{args.results_folder}/train_codes.pickle', 'rb') as f:
        train_codes = pickle.load(f)

    with open(f'{args.results_folder}/valid_codes.pickle', 'rb') as f:
        valid_codes = pickle.load(f)

    # get codes for random noisy images and sub_folders of attacked classifiers
    bounds = []
    noisy_codes = []
    subfolders = []

    files = sorted(os.listdir(args.results_folder))
    for f in files:
        if f.startswith('valid') and 'l2=' in f:
            bounds.append(f.split('l2=')[1].replace('.pickle', ''))

            with open(f'{args.results_folder}/{f}', 'rb') as nf:
                noisy_codes.append(pickle.load(nf))

        elif os.path.isdir(f'{args.results_folder}/{f}'):
            subfolders.append(f'{args.results_folder}/{f}')

    # compute KL divergences for random noise
    latents = list(train_codes.keys())[::-1]
    kl_matrix = torch.zeros((len(latents), len(noisy_codes) + 1))

    for idx_latent in latents:

        # get P distribution from train set.
        train_dist = torch.tensor(train_codes[idx_latent])
        P_distribution = torch.distributions.MultivariateNormal(train_dist.mean(dim=0),
                                                                covariance_matrix=torch.diag(train_dist.var(dim=0)))

        for col, test_codes in enumerate([valid_codes] + noisy_codes):

            # get Q (test distribution)
            test_dist = torch.tensor(test_codes[idx_latent])
            Q_distribution = torch.distributions.MultivariateNormal(test_dist.mean(dim=0),
                                                                    covariance_matrix=torch.diag(test_dist.var(dim=0)))

            kl_divergence = torch.distributions.kl_divergence(P_distribution, Q_distribution).item()
            kl_matrix[len(latents) - idx_latent - 1, col] = kl_divergence

    # plot matrix for random noise
    fig = plt.figure(figsize=(4, 0.3 * len(latents) + 2))
    plt.imshow(kl_matrix.cpu().numpy(), cmap='viridis', interpolation='nearest', aspect='auto')
    plt.grid(True, which='both', linestyle='-', linewidth=1, color='black')

    plt.xlabel('L2 Bound')

    # set x and y ticks with labels centered on cells (not on ticks)
    xticks = np.arange(-0.5, len(bounds) + 0.5, 1)
    yticks = np.arange(-0.5, len(latents) - 0.5, 1)
    plt.xticks(xticks, ['' for _ in xticks])
    plt.yticks(yticks, ['' for _ in yticks])

    for tick, label in zip(xticks, ['0.00'] + bounds):
        plt.text(tick + 0.5, 1.0 + kl_matrix.shape[0], label, ha='center', va='center', fontsize=10)

    for tick, label in zip(yticks, latents):
        plt.text(-0.75, tick + 0.5, label, ha='center', va='center', fontsize=10)

    plt.colorbar(label='KL divergence')

    # Add value annotations
    for i in range(kl_matrix.shape[0]):
        for j in range(kl_matrix.shape[1]):
            plt.text(j, i, f'{kl_matrix[i, j]:.2f}', ha='center', va='center', color='black')

    plt.suptitle(f'KL divergence per latent on {args.autoencoder_name}\nwith Random Gaussian Noise')
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    # ####################################################################################################

    # get codes for each attack
    attacks = {
        # classifier_attack : {bounds: list, success_rates: list, codes: list }
    }

    for classifier_ in subfolders:

        base_key_name = classifier_.split('/')[-1]

        attack_files = sorted(os.listdir(classifier_))
        for f in attack_files:

            attack_name = f.split('_')[0]
            key_name = base_key_name + '_' + attack_name

            if key_name not in attacks.keys():
                attacks[key_name] = {'bounds': [], 'success_rates': [], 'codes': []}

            if f.endswith('.pickle'):

                # add bound to dict
                attacks[key_name]['bounds'].append(f.split('l2=')[1].replace('.pickle', ''))

                with open(f'{classifier_}/{f}', 'rb') as nf:
                    attacks[key_name]['codes'].append(pickle.load(nf))

            else:
                # .npy success rates
                sr = np.load(f'{classifier_}/{f}')
                attacks[key_name]['success_rates'].append(sr)

    # compute KL and plot each attack
    for net_attack in attacks.keys():

        bounds = attacks[net_attack]['bounds']
        success_rates = attacks[net_attack]['success_rates']
        codes = attacks[net_attack]['codes']
        classifier_name, attack_name = net_attack.split('_')

        # compute KL divergences for random noise
        latents = list(train_codes.keys())[::-1]
        kl_matrix = torch.zeros((len(latents), len(codes) + 1))

        for idx_latent in latents:

            # get P distribution from train set.
            train_dist = torch.tensor(train_codes[idx_latent])
            P_distribution = torch.distributions.MultivariateNormal(train_dist.mean(dim=0),
                                                                    covariance_matrix=torch.diag(train_dist.var(dim=0)))

            for col, test_codes in enumerate([valid_codes] + codes):
                # get Q (test distribution)
                test_dist = torch.tensor(test_codes[idx_latent])
                Q_distribution = torch.distributions.MultivariateNormal(test_dist.mean(dim=0),
                                                                        covariance_matrix=torch.diag(
                                                                            test_dist.var(dim=0)))

                kl_divergence = torch.distributions.kl_divergence(P_distribution, Q_distribution).item()
                kl_matrix[len(latents) - idx_latent - 1, col] = kl_divergence

        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8, 0.3 * len(latents) + 2))

        xticks = np.arange(len(bounds))
        yticks = np.arange(0.0, 1.1, 0.1)
        ax[0].set_xticks(xticks, bounds)
        ax[0].set_xlabel('L2 Bound')
        ax[0].set_yticks(yticks, [f'{y:.1f}' for y in yticks])
        ax[0].set_ylabel('Success Rate')
        success_rates = [np.mean(s) for s in success_rates]
        ax[0].plot(success_rates)

        im = ax[1].imshow(kl_matrix.cpu().numpy(), cmap='viridis', interpolation='nearest', aspect='auto')
        ax[1].grid(True, which='both', linestyle='-', linewidth=1, color='black')

        ax[1].set_xlabel('L2 Bound')

        # set x and y ticks with labels centered on cells (not on ticks)
        xticks = np.arange(-0.5, len(bounds) + 0.5, 1)
        yticks = np.arange(-0.5, len(latents) - 0.5, 1)
        ax[1].set_xticks(xticks, ['' for _ in xticks])
        ax[1].set_yticks(yticks, ['' for _ in yticks])

        for tick, label in zip(xticks, ['0.00'] + bounds):
            ax[1].text(tick + 0.5, 1.0 + kl_matrix.shape[0], label, ha='center', va='center', fontsize=10)

        for tick, label in zip(yticks, latents):
            ax[1].text(-0.75, tick + 0.5, label, ha='center', va='center', fontsize=10)

        cbar = plt.colorbar(im, ax=ax[1])
        cbar.set_label('KL Divergence')

        # Add value annotations
        for i in range(kl_matrix.shape[0]):
            for j in range(kl_matrix.shape[1]):
                ax[1].text(j, i, f'{kl_matrix[i, j]:.2f}', ha='center', va='center', color='black')

        plt.suptitle(f'{attack_name} L2 Attack on {classifier_name} success rate and KL divergence per latent')
        plt.tight_layout()
        plt.show()
        plt.close(fig)


if __name__ == '__main__':

    arguments = parse_args()

    main(arguments)
