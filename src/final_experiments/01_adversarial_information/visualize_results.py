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

    # cnn name and generator name
    adversarial_pickles_folder = adversarial_pickles_folder if adversarial_pickles_folder[-1] != '/' \
        else adversarial_pickles_folder[:-1]

    distances_pickles_folder = distances_pickles_folder if distances_pickles_folder[-1] != '/' \
        else distances_pickles_folder[:-1]

    cnn_name = adversarial_pickles_folder.split('/')[-1].split('_')[-1]
    gen_name = distances_pickles_folder.split('/')[-2]

    # dataloader and batch_size random indices
    image_size = 32 if 'cifar10' in images_folder else '256'
    data = ImageDataset(folder=images_folder, image_size=image_size)
    indexes = np.random.randint(low=0, high=len(data), size=batch_size)
    gt_images = np.stack([data[i] for i in indexes], axis=0).transpose(0, 2, 3, 1)

    # make dir to save plots
    save_dir = f'{adversarial_pickles_folder}/plots/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # get all pickle files
    attack_files = sorted(os.listdir(adversarial_pickles_folder))[:-1]  # remove the created 'plots' folder
    distances_files = sorted(os.listdir(distances_pickles_folder))

    final_dict = {}

    for (a_f, d_f) in zip(attack_files, distances_files):

        # check that we are opening results for the same file!
        assert a_f == d_f

        # get files content
        with open(f'{adversarial_pickles_folder}/{a_f}', 'rb') as f:
            adversaries = pickle.load(f)

        with open(f'{distances_pickles_folder}/{d_f}', 'rb') as f:
            latents = pickle.load(f)

        # # load and save some adversaries
        # adversarial_examples = np.stack([adversaries['adversarial_samples'][f'{i}'] for i in indexes], axis=0)
        # adversarial_diffs = np.abs(gt_images - adversarial_examples)
        # norm_factor = np.max(adversarial_diffs.reshape(batch_size, -1), axis=1).reshape(batch_size, 1, 1, 1)
        # adversarial_diffs = adversarial_diffs / norm_factor
        #
        # display = np.concatenate([gt_images, adversarial_examples, adversarial_diffs]).transpose(0, 3, 1, 2)
        # display = make_grid(torch.tensor(display), nrow=batch_size).permute(1, 2, 0).numpy()
        # plt.imshow(display)
        # plt.axis(False)
        # plt.title(f'Attack: {a_f[:-7]} on {cnn_name}')
        # plt.tight_layout()
        # plt.savefig(f'{save_dir}/{a_f[:-7]}.png')
        # plt.close()

        # update final dict
        attack_name, bound_n = a_f[:-7].split('=')

        if attack_name not in final_dict.keys():
            final_dict[attack_name] = {}

        final_dict[attack_name][bound_n] = {
            'success_rate': adversaries['success_rate'],
        }
        final_dict[attack_name][bound_n].update(latents)

    print(f'ATTACKS TO {cnn_name} MEASURED ON {gen_name}')
    for attack_name in final_dict.keys():

        print('#'*25)
        print(f'{attack_name}')

        bounds = []
        success_rates = []
        rows = None

        for bound_n in final_dict[attack_name].keys():

            if rows is None:
                rows = [f'{l} &' for l in range(len(final_dict[attack_name][bound_n]) - 1)]  # n latents

            bounds.append(bound_n)
            success_rates.append(final_dict[attack_name][bound_n]['success_rate'])

            for latent in final_dict[attack_name][bound_n].keys():
                if latent == 'success_rate':
                    continue

                latent_num = int(latent.split("_")[1])
                rows[latent_num] += f' {final_dict[attack_name][bound_n][latent]["kl_divergence"]:.3f} &'
                rows[latent_num] += f' {final_dict[attack_name][bound_n][latent]["l2_mean"]:.3f} &'
                rows[latent_num] += f' {final_dict[attack_name][bound_n][latent]["l2_std"]:.3f} & '

        print('bounds:', bounds)

        for r in rows:
            print(r[:-1] + '\\\\')

        # plt.plot(bounds, success_rates)
        # plt.tight_layout(pad=2.5)
        # plt.title(f'attack: {attack_name} on {cnn_name}')
        # plt.xlabel('bound value')
        # plt.ylabel('attack success rate')
        # plt.savefig(f'{save_dir}/{attack_name}_success.png')
        # plt.close()

if __name__ == '__main__':

    arguments = parse_args()

    main(adversarial_pickles_folder=arguments.adversarial_results,
         distances_pickles_folder=arguments.latent_distances_results,
         images_folder=arguments.images_folder,
         batch_size=arguments.batch_size)
