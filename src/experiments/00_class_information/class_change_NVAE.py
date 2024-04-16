import argparse
from einops import rearrange
from kornia.enhance import normalize
from matplotlib import pyplot as plt
import numpy as np
import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from data.datasets import CoupledDataset
from src.hl_autoencoders.NVAE.mine.distributions import DiscMixLogistic
from src.defenses.common_utils import load_NVAE, load_hub_CNN


def parse_args():

    parser = argparse.ArgumentParser('Test top-1 accuracy of CNNs when interpolating images on NVAE chunks')

    # TODO for public release, replace "default=..." with "required=True"

    parser.add_argument('--data_path', type=str, default='/media/dserez/datasets/cifar10/validation/',
                        help='directory of validation set')

    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--cnn_type', type=str, required=True, choices=['resnet32', 'vgg16'])

    parser.add_argument('--cnn_path', type=str, default='/media/dserez/runs/adversarial/CNNs/',
                        help='directory of validation set')

    parser.add_argument('--nvae_path', type=str, required=True,
                        help='checkpoint file for NVAE, containing state dict and configuration information')

    parser.add_argument('--name', type=str, default='test',
                        help='name of experiments for saving results')

    parser.add_argument('--visual_examples_saving_folder', type=str, default='./plots/',
                        help='where to save visual examples of the reconstructed interpolations')

    parser.add_argument('--pickle_file_saving_folder', type=str, default='./results/',
                        help='where to save pickle file with the accuracies')

    args = parser.parse_args()

    # create dirs if not existing
    if not os.path.exists(f'{args.visual_examples_saving_folder}/{args.name}'):
        os.makedirs(f'{args.visual_examples_saving_folder}/{args.name}')

    if not os.path.exists(f'{args.pickle_file_saving_folder}/{args.name}'):
        os.makedirs(f'{args.pickle_file_saving_folder}/{args.name}')

    return args


@torch.no_grad()
def main(data_path: str, batch_size: int, cnn_type: str, cnn_path: str, nvae_path: str,
         plots_dir: str, pickle_dir: str):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # this is used for loading validation in consistent order, allowing nice plots.
    random_seed = 0
    torch.manual_seed(random_seed)

    # load dataset
    dataloader = DataLoader(CoupledDataset(folder=data_path, image_size=32, seed=random_seed),
                            batch_size=batch_size, shuffle=False)

    # load pretrained NVAE
    nvae = load_NVAE(nvae_path, device)
    n_latents = sum(nvae.groups_per_scale)

    # load pretrained classifier
    cnn = load_hub_CNN(cnn_path, cnn_type, device)

    latents_to_test = ['all'] + np.arange(n_latents).tolist()
    alphas = np.arange(0, 1.1, 0.1)

    # dict for saving
    dict_accuracies = {}  # will be {i : {a: accuracy, ...} , ... } where i = latent_idx and a = alpha

    n_samples = 10   # to visualize
    example_images = torch.zeros((n_samples, len(latents_to_test), len(alphas), 3, 32, 32), device='cpu')

    print('[INFO] Starting Computation on each latent...')

    for latent_idx, latent in enumerate(tqdm(latents_to_test)):

        for alpha_idx, a in enumerate(alphas):

            all_preds = torch.empty(0, device=device)
            all_labels = torch.empty(0, device=device)

            for batch_idx, batch in enumerate(dataloader):

                # first images and labels, to_interpolate
                x1, y1, x2, _ = [i.to(device) for i in batch]

                # encode
                chunks_x1 = nvae.encode(x1, deterministic=True)
                chunks_x2 = nvae.encode(x2, deterministic=True)

                # interpolate latent i at alpha
                if latent == 'all':
                    interpolated_chunks = [(1 - a) * c1 + a * c2 for (c1, c2) in zip(chunks_x1, chunks_x2)]
                else:
                    interpolated_chunks = chunks_x1.copy()
                    interpolated_chunks[latent] = (1 - a) * interpolated_chunks[latent] + a * chunks_x2[latent]

                # decode batch
                logits = nvae.decode(interpolated_chunks)
                recons = DiscMixLogistic(logits).mean()

                # save image for later
                if batch_idx < n_samples:
                    example_images[batch_idx, latent_idx, alpha_idx] = recons[0].cpu()

                # get prediction
                recons = normalize(recons,
                                   mean=torch.tensor([0.507, 0.4865, 0.4409], device=device),
                                   std=torch.tensor([0.2673, 0.2564, 0.2761], device=device))
                preds = torch.argmax(cnn(recons), dim=1)

                # save preds, targets of batch
                all_preds = torch.cat((all_preds, preds))
                all_labels = torch.cat((all_labels, y1))

            # save to final dict
            accuracy = (all_preds == all_labels).to(torch.float32).mean().item()

            if str(latent) not in dict_accuracies.keys():
                dict_accuracies[str(latent)] = {f'{a:.2f}': accuracy}
            else:
                dict_accuracies[str(latent)][f'{a:.2f}'] = accuracy

    # save image
    for i, sample in enumerate(example_images):

        display = make_grid(rearrange(sample, 'l a c h w -> (l a) c h w'), nrow=len(alphas))
        display = rearrange(display, 'c h w -> h w c').numpy()
        plt.imshow(display)
        plt.axis(False)
        plt.savefig(f'{plots_dir}/chunks_interpolation_sample_{i}.png')
        plt.close()

    # save pickle
    with open(f'{pickle_dir}/class_change_accuracies.json', 'w') as f:
        json.dump(dict_accuracies, f)


if __name__ == '__main__':

    arguments = parse_args()

    main(data_path=arguments.data_path,
         batch_size=arguments.batch_size,
         cnn_type=arguments.cnn_type,
         cnn_path=arguments.cnn_path,
         nvae_path=arguments.nvae_path,
         plots_dir=f'{arguments.visual_examples_saving_folder}/{arguments.name}',
         pickle_dir=f'{arguments.pickle_file_saving_folder}/{arguments.name}',
         )
