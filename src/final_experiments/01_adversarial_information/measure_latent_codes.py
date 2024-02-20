import argparse
import os
import pickle

import torch
from einops import rearrange
from kornia.enhance import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from data.datasets import ImageDataset
from src.final_experiments.common_utils import load_StyleGan, load_NVAE

"""
IMPORTANT! This script assumes that you have already run "compute_attacks.py" and 
           that .pickle files are stored as there described (same folder structure)
"""

def parse_args():

    parser = argparse.ArgumentParser('Measure and Save Latent Space Codes for different precomputed Attacks')

    parser.add_argument('--attacks_folder', type=str, required=True,
                        help='path to folder containing .pickle files of precomputed attacks.\n'
                             ' NOTE: results folder is automatically saved in the same directory than --attacks_folder')

    parser.add_argument('--images_folder', type=str, required=True,
                        help='Same set of images used for the attacks')

    parser.add_argument('--batch_size', type=int, required=True)

    parser.add_argument('--autoencoder_path', type=str, required=True,
                        help='path to the pre-trained generator/autoencoder model')

    args = parser.parse_args()

    return args


@torch.no_grad()
def main(attacks_folder: str, images_folder: str, batch_size: int, autoencoder_path: str, device: str):

    # derive cnn_name from attacks_folder
    if attacks_folder[-1] == '/':
        attacks_folder = attacks_folder[:-1]

    cnn_name = attacks_folder.split("/")[-1].split("_")[-1]

    # get all pickle files (will maintain same names for saving)
    all_pickle_files = os.listdir(attacks_folder)

    # load correct network
    if cnn_name == "resnet-50":
        autoencoder = load_StyleGan(autoencoder_path, device)
        image_size = 256
        net_name = 'StyleGan'

    else:
        autoencoder = load_NVAE(autoencoder_path, device)
        image_size = 32
        net_name = f'NVAE_{autoencoder_path.split("/")[-1].split(".")[0]}'

    # create destination folder for later
    dst_folder = '/'.join(attacks_folder.split("/")[:-1]) + f'/{net_name}/latent_distances_{cnn_name}'
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # dataloader
    dataloader = DataLoader(ImageDataset(folder=images_folder, image_size=image_size),
                            batch_size=batch_size, shuffle=False)

    # clean_samples is formatted as : {'l': {'i': np.array; ...}; ... }
    # where 'i' is the sample num, 'l' is the latent index and np.array is the code of len(dim).
    # final_dict is formatted as: {attack_n: {'bound_n': {'l': {'i': np.array; ...}; ... }; ...}
    clean_samples = {}
    final_dict = {}

    for f_num, pickle_name in enumerate(tqdm(all_pickle_files)):

        # load pre-computed adversaries
        pickle_file = f'{attacks_folder}/{pickle_name}'
        with open(pickle_file, 'rb') as f:
            attack_results = pickle.load(f)

        # create entry in final dict
        attack_name = pickle_name.split('.')[0]
        final_dict[attack_name] = {}

        for batch_idx, samples in enumerate(dataloader):

            # if batch_idx == 8:
            #     break

            samples = samples.to(device)

            # compute and save clean samples
            if f_num == 0:  # DO ONLY FIRST TIME (CLEAN SAMPLES DON'T CHANGE)
                if cnn_name == 'resnet-50':
                    samples = normalize(samples,
                              torch.tensor([0.5, 0.5, 0.5], device=device),
                              torch.tensor([0.5, 0.5, 0.5], device=device))
                    clean_codes = autoencoder.encode(samples)
                    pattern = ' '  # TODO
                else:
                    clean_codes = autoencoder.encode(samples, deterministic=True)
                    pattern = 'd h w -> (d h w)'

                for latent_idx, latents in enumerate(clean_codes):

                    if latent_idx not in clean_samples.keys():
                        clean_samples[latent_idx] = {}

                    for img_idx, c in enumerate(latents):

                        sample_n = f'{int(batch_idx * batch_size) + img_idx}'
                        clean_samples[latent_idx][sample_n] = rearrange(c, pattern).cpu().numpy()

            # Compute and save codes for each attack-bound
            for bound in attack_results.keys():

                # get adversaries and prepare dict
                adversaries = attack_results[bound]
                final_dict[attack_name][bound] = {}

                # load adversaries
                advs = np.stack([adversaries[str(i)][0] for i in
                                 range(batch_idx * batch_size, (batch_idx + 1) * batch_size)])
                advs = torch.from_numpy(advs).to(device)

                if cnn_name == 'resnet-50':
                    advs = normalize(advs,
                                     torch.tensor([0.5, 0.5, 0.5], device=device),
                                     torch.tensor([0.5, 0.5, 0.5], device=device))
                    advs_codes = autoencoder.encode(advs)
                    pattern = ' '  # TODO
                else:
                    advs_codes = autoencoder.encode(advs, deterministic=True)
                    pattern = 'd h w -> (d h w)'

                for latent_idx, latents in enumerate(advs_codes):

                    if latent_idx not in final_dict[attack_name][bound].keys():
                        final_dict[attack_name][bound][latent_idx] = {}

                    for img_idx, c in enumerate(latents):
                        sample_n = f'{int(batch_idx * batch_size) + img_idx}'
                        final_dict[attack_name][bound][latent_idx][sample_n] = rearrange(c, pattern).cpu().numpy()

    # save pickle file for clean samples
    with open(f'{dst_folder}/CleanSamples.pickle', 'wb') as f:
        pickle.dump(clean_samples, f)

    for attack_name in final_dict.keys():

        with open(f'{dst_folder}/{attack_name}.pickle', 'wb') as f:
            pickle.dump(final_dict[attack_name], f)


if __name__ == '__main__':

    arguments = parse_args()

    main(attacks_folder=arguments.attacks_folder,
         images_folder=arguments.images_folder,
         batch_size=arguments.batch_size,
         autoencoder_path=arguments.autoencoder_path,
         device='cuda:0'
         )


