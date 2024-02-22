import argparse

import torch
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle

from torchvision.utils import make_grid

from data.datasets import ImageDataset


def parse_args():

    parser = argparse.ArgumentParser('Visualize results of "01_adversarial_information" experiments')

    parser.add_argument('--adversarial_results', type=str,
                        required=True, help='folder with .pickle files of pre-computed attacks')

    parser.add_argument('--latent_distances_results', type=str,
                        required=True, help='folder with .pickle files of pre-computed latent_distances')

    parser.add_argument('--images_folder', type=str, required=True,
                        help='same set of images used for the computations')

    parser.add_argument('--batch_size', type=int, required=True)

    args = parser.parse_args()

    # check file exists
    if not os.path.exists(args.adversarial_results):
        raise FileNotFoundError(f'File not Found: {args.adversarial_results}')

    if not os.path.exists(args.latent_distances_results):
        raise FileNotFoundError(f'File not Found: {args.latent_distances_results}')

    if not os.path.exists(args.images_folder):
        raise FileNotFoundError(f'File not Found: {args.images_folder}')

    return args


def main(adversarial_pickles_folder: str, distances_pickles_folder: str, images_folder: str, batch_size: int):

    # extract cnn name and generator name for saving results
    adversarial_pickles_folder = adversarial_pickles_folder if adversarial_pickles_folder[-1] != '/' \
        else adversarial_pickles_folder[:-1]

    distances_pickles_folder = distances_pickles_folder if distances_pickles_folder[-1] != '/' \
        else distances_pickles_folder[:-1]

    cnn_name = adversarial_pickles_folder.split('/')[-1].split('_')[-1]
    gen_name = distances_pickles_folder.split('/')[-2]

    # create dir to save plots
    save_dir = f'./plots/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save an out file at the end
    out_file_name = f'./{cnn_name}_{gen_name}_table_results.txt'
    printed_rows = []

    # load a random batch of gt samples.
    image_size = 32 if 'cifar10' in images_folder else 256
    data = ImageDataset(folder=images_folder, image_size=image_size)
    random_indexes = np.random.randint(low=0, high=len(data), size=batch_size)
    gt_images = np.stack([data[i] for i in random_indexes], axis=0)

    # get all pickle files (in the same ordering, since they have the same name)
    attack_files = sorted(os.listdir(adversarial_pickles_folder))
    distances_files = sorted(os.listdir(distances_pickles_folder))
    distances_files.remove('CleanSamples.pickle')

    # load clean samples, formatted as: {latent_idx (int): {'sample_idx' (str): np.array (latent_len)}}
    with open(f'{distances_pickles_folder}/CleanSamples.pickle', 'rb') as f:
        clean_latents = pickle.load(f)

    # from 0 to n_latents
    arange_on_clean_latents = list(clean_latents.keys())

    # loop all files
    for (attack_f, distances_f) in zip(attack_files, distances_files):

        # check that we are opening results for the same file!
        assert attack_f == distances_f

        # get files content
        with open(f'{adversarial_pickles_folder}/{attack_f}', 'rb') as f:

            # Formatted as {bound (str): {sample_idx (str): [np.array (latent_len), success: float]}}
            adversaries_and_success_rate_dict = pickle.load(f)

        with open(f'{distances_pickles_folder}/{distances_f}', 'rb') as f:

            # Formatted as {bound (str): {latent_idx (int), sample_idx (str): np.array (latent_len)}}
            adversarial_latents = pickle.load(f)

        # get list of bounds for this attack
        bounds = list(adversaries_and_success_rate_dict.keys())
        overall_success_rate_per_bound = []  # save overall success rate for final plot

        printed_rows.append('########################## \n')
        printed_rows.append(f'Attack & Latent & {" ".join(bounds)} \\\\ \n')

        # each row is formatted as  "attack name & latent idx & KL_for_bound ... "
        latents_vs_bounds_rows = [f'{attack_f[:-7]} & {idx} & ' for idx in arange_on_clean_latents]

        # same but KL is computed only in the successful samples
        latents_vs_bounds_rows_successful = [f'{attack_f[:-7]}-S & {idx} & ' for idx in arange_on_clean_latents]

        # loop on each bound
        for bound_idx, bound in enumerate(bounds):

            # load and save some adversaries
            adversarial_examples = np.stack([adversaries_and_success_rate_dict[bound][f'{i}'][0] for i in random_indexes], axis=0)
            adversarial_diffs = np.abs(gt_images - adversarial_examples)
            norm_factor = np.max(adversarial_diffs.reshape(batch_size, -1), axis=1).reshape(batch_size, 1, 1, 1)
            adversarial_diffs = adversarial_diffs / norm_factor

            display = np.concatenate([gt_images, adversarial_examples, adversarial_diffs])
            display = make_grid(torch.tensor(display), nrow=batch_size).permute(1, 2, 0).numpy()
            plt.imshow(display)
            plt.axis(False)
            plt.title(f'Attack: {attack_f[:-7]}={bound} on {cnn_name}')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/{attack_f[:-7]}={bound}_{cnn_name}_{gen_name}_adversaries.png')
            plt.close()

            # load vector of success rate (1, 0) and save the overall success rate
            attack_success = np.stack([adversaries_and_success_rate_dict[bound][f'{i}'][1] for i in range(len(data))], axis=0)
            attack_success = torch.tensor(attack_success)
            overall_success_rate_per_bound.append(torch.mean(attack_success).item())

            for latent_idx, latent in enumerate(arange_on_clean_latents):

                # get adversarial latent vectors.
                cln = np.stack([clean_latents[latent][f'{i}'] for i in range(len(data))], axis=0)
                cln = torch.tensor(cln)
                adv = np.stack([adversarial_latents[bound][latent][f'{i}'] for i in range(len(data))], axis=0)
                adv = torch.tensor(adv)

                # TODO what changes if you take the norm ?


                # compute KL distance for all vs only successful attacks
                # L2 TODO ensure this is correct
                # l2_dist = torch.norm(a / c, dim=1, p=2)
                # l2_dist = l2_dist / c.shape[1]  # normalize by size
                # l2_mean = l2_dist.mean().item()
                # l2_std = l2_dist.std().item()

                # KL ALL
                # Create distributions from the data
                P_distribution = torch.distributions.MultivariateNormal(cln.mean(dim=0),
                                                                        covariance_matrix=torch.diag(cln.var(dim=0)))
                Q_distribution = torch.distributions.MultivariateNormal(adv.mean(dim=0),
                                                                        covariance_matrix=torch.diag(adv.var(dim=0)))

                # Compute the KL divergence
                kl_divergence = torch.distributions.kl_divergence(P_distribution, Q_distribution).item()
                latents_vs_bounds_rows[latent_idx] += f'{kl_divergence:.2f} & '

                # KL SUCCESSFUL ONLY
                cln = cln[attack_success.bool()]
                adv = adv[attack_success.bool()]

                # Create distributions from the data
                P_distribution = torch.distributions.MultivariateNormal(cln.mean(dim=0),
                                                                        covariance_matrix=torch.diag(cln.var(dim=0)))
                Q_distribution = torch.distributions.MultivariateNormal(adv.mean(dim=0),
                                                                        covariance_matrix=torch.diag(adv.var(dim=0)))

                # Compute the KL divergence
                kl_divergence = torch.distributions.kl_divergence(P_distribution, Q_distribution).item()
                latents_vs_bounds_rows_successful[latent_idx] += f'{kl_divergence:.2f} & '



        printed_rows += [f'{row[:-2]} \\\\ \n' for row in latents_vs_bounds_rows]
        printed_rows += [f'{row[:-2]} \\\\ \n' for row in latents_vs_bounds_rows_successful]

        # plot success rates
        plt.plot(bounds, overall_success_rate_per_bound)
        plt.tight_layout(pad=2.5)
        plt.title(f'attack: {attack_f[:-7]} on {cnn_name}')
        plt.xlabel('bound value')
        plt.ylabel('attack success rate')
        plt.savefig(f'{save_dir}/{attack_f[:-7]}_{cnn_name}_{gen_name}_success_rates.png')
        plt.close()

    with open(out_file_name, 'w') as f:
        f.writelines(printed_rows)

if __name__ == '__main__':

    arguments = parse_args()

    main(adversarial_pickles_folder=arguments.adversarial_results,
         distances_pickles_folder=arguments.latent_distances_results,
         images_folder=arguments.images_folder,
         batch_size=arguments.batch_size)
