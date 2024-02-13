import os
import pickle

from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import ImageDataset
from src.final_experiments.common_utils import load_StyleGan, load_NVAE

"""
IMPORTANT! This script assumes that you have already run "compute_attacks.py" and 
           that .pickle files are stored as there described (same folder structure)
"""


def main(attacks_folder: str, images_path: str, batch_size:int,
         decoder_path: str, encoder_path: str or None, device: str):

    # derive paths from attacks_folder
    cnn_name = attacks_folder.split("/")[-1].split("_")[-1]
    dst_folder = f'latent_distances_{cnn_name}'
    all_pickle_files = os.listdir(attacks_folder)

    # load correct network
    if cnn_name == "resnet-50":
        autoencoder = load_StyleGan(encoder_path, decoder_path, device)
        image_size = 256
    else:
        autoencoder = load_NVAE(decoder_path, device)
        image_size = 32

    # dataloader
    dataloader = DataLoader(ImageDataset(folder=images_path, image_size=image_size),
                            batch_size=batch_size, shuffle=False)

    for pickle_name in tqdm(all_pickle_files):

        pickle_file = attacks_folder + pickle_name
        with open(pickle_file, 'rb') as f:
            attack_results = pickle.load(f)

        adversaries = attack_results['adversaries']

        for batch_idx, samples in enumerate(dataloader):

            a = [adversaries[str(i)] for i in range(batch_idx * batch_size, (batch_idx + 1) * batch_size)]


if __name__ == '__main__':

    main(attacks_folder='/media/dserez/runs/adversarial/01_adversarial_information/attacks_to_resnet-50/')