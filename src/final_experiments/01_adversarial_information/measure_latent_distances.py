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

    parser = argparse.ArgumentParser('Measure Latent Space Distances for different precomputed Attacks')

    parser.add_argument('--attacks_folder', type=str, required=True,
                        help='path to folder containing .pickle files of precomputed attacks.\n'
                             ' NOTE: results folder is automatically saved in the same directory than --attacks_folder')

    parser.add_argument('--images_folder', type=str, required=True,
                        help='Same set of images used for the attacks')

    parser.add_argument('--batch_size', type=int, required=True)

    parser.add_argument('--decoder_path', type=str, required=True,
                        help='path to the pre-trained generator model')

    parser.add_argument('--encoder_path', type=str, default=None,
                        help='path to the pre-trained encoder model (if any)')

    args = parser.parse_args()

    return args


@torch.no_grad()
def main(attacks_folder: str, images_folder: str, batch_size: int,
         decoder_path: str, encoder_path: str or None, device: str):

    # derive cnn_name from attacks_folder
    if attacks_folder[-1] == '/':
        attacks_folder = attacks_folder[:-1]

    cnn_name = attacks_folder.split("/")[-1].split("_")[-1]

    # get all pickle files (will maintain same names for saving)
    all_pickle_files = os.listdir(attacks_folder)

    # load correct network
    if cnn_name == "resnet-50":
        autoencoder = load_StyleGan(encoder_path, decoder_path, device)
        image_size = 256
        net_name = 'StyleGan'  # TODO

    else:
        autoencoder = load_NVAE(decoder_path, device)
        image_size = 32
        net_name = f'NVAE_{decoder_path.split("/")[-1].split(".")[0]}'

    # create destination folder for later
    dst_folder = '/'.join(attacks_folder.split("/")[:-1]) + f'/{net_name}/latent_distances_{cnn_name}'
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # dataloader
    dataloader = DataLoader(ImageDataset(folder=images_folder, image_size=image_size),
                            batch_size=batch_size, shuffle=False)

    # obtain these only once!
    clean_dataset_codes = []

    for f_num, pickle_name in enumerate(tqdm(all_pickle_files)):

        # load pre-computed adversaries
        pickle_file = f'{attacks_folder}/{pickle_name}'
        with open(pickle_file, 'rb') as f:
            attack_results = pickle.load(f)

        adversaries = attack_results['adversarial_samples']

        # these need to be obtained each time
        adver_dataset_codes = []

        for batch_idx, samples in enumerate(dataloader):

            # if batch_idx == 8:
            #     break

            samples = samples.to(device)

            advs = np.stack([adversaries[str(i)] for i in range(batch_idx * batch_size, (batch_idx + 1) * batch_size)])
            advs = torch.from_numpy(advs).to(device).permute(0, 3, 1, 2)

            # get latent codes
            if cnn_name == 'resnet-50':

                # STYLEGAN CASE
                samples = normalize(samples,
                                    torch.tensor([0.5, 0.5, 0.5], device=device),
                                    torch.tensor([0.5, 0.5, 0.5], device=device))
                advs = normalize(advs,
                                 torch.tensor([0.5, 0.5, 0.5], device=device),
                                 torch.tensor([0.5, 0.5, 0.5], device=device))

                # get codes for clean case (only first time)
                if f_num == 0:

                    # always use standard e4e autoencoder (hyperstyle weights are applied only when decoding)
                    _, clean_codes = autoencoder.get_initial_inversion(samples)
                    clean_codes = rearrange(clean_codes, 'b n d -> n b d')  # n latents at first

                    for code_idx, code in enumerate(clean_codes):

                        if len(clean_dataset_codes) < len(clean_codes):
                            clean_dataset_codes.append(code)
                        else:
                            clean_dataset_codes[code_idx] = torch.cat(
                                [clean_dataset_codes[code_idx], code], dim=0
                            )

                # do the same for adversaries (changes at each file)
                _, adver_codes = autoencoder.get_initial_inversion(advs)
                adver_codes = rearrange(adver_codes, 'b n d -> n b d')  # n latents at first

                for code_idx, code in enumerate(adver_codes):

                    if len(adver_dataset_codes) < len(adver_codes):
                        adver_dataset_codes.append(code)
                    else:
                        adver_dataset_codes[code_idx] = torch.cat(
                            [adver_dataset_codes[code_idx], code], dim=0
                        )

            else:

                # NVAE CASE

                # clean only first time
                if f_num == 0:
                    clean_codes = autoencoder.encode(samples, deterministic=True)
                    for code_idx, code in enumerate(clean_codes):

                        code = rearrange(code, 'b c h w -> b (c h w)')

                        if len(clean_dataset_codes) < len(clean_codes):
                            clean_dataset_codes.append(code)
                        else:
                            clean_dataset_codes[code_idx] = torch.cat(
                                [clean_dataset_codes[code_idx], code], dim=0
                            )

                # do the same for adversaries (changes at each file)
                adver_codes = autoencoder.encode(advs, deterministic=True)

                for code_idx, code in enumerate(adver_codes):

                    code = rearrange(code, 'b c h w -> b (c h w)')

                    if len(adver_dataset_codes) < len(adver_codes):
                        adver_dataset_codes.append(code)
                    else:
                        adver_dataset_codes[code_idx] = torch.cat(
                            [adver_dataset_codes[code_idx], code], dim=0
                        )

        # compute l2 and kl divergence on all dataset
        this_pickle = {}
        for i, (a, c) in enumerate(zip(adver_dataset_codes, clean_dataset_codes)):
            # L2 TODO ensure this is correct
            l2_dist = torch.norm(a / c, dim=1, p=2)
            l2_dist = l2_dist / c.shape[1]  # normalize by size
            l2_mean = l2_dist.mean().item()
            l2_std = l2_dist.std().item()

            # KL
            # Create distributions from the data
            P_distribution = torch.distributions.MultivariateNormal(c.mean(dim=0),
                                                                    covariance_matrix=torch.diag(c.var(dim=0)))
            Q_distribution = torch.distributions.MultivariateNormal(a.mean(dim=0),
                                                                    covariance_matrix=torch.diag(a.var(dim=0)))

            # Compute the KL divergence
            kl_divergence = torch.distributions.kl_divergence(P_distribution, Q_distribution).item()

            this_pickle[f'latent_{i}'] = {
                'l2_mean': l2_mean,
                'l2_std': l2_std,
                'kl_divergence': kl_divergence
            }

        # save pickle file for this attack
        with open(f'{dst_folder}/{pickle_name}', 'wb') as f:
            pickle.dump(this_pickle, f)

if __name__ == '__main__':

    arguments = parse_args()

    main(attacks_folder=arguments.attacks_folder,
         images_folder=arguments.images_folder,
         batch_size=arguments.batch_size,
         decoder_path=arguments.decoder_path,
         encoder_path=arguments.encoder_path,
         device='cuda:0'
         )


