import argparse
import os
import pickle

import torch
from einops import rearrange
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from kornia.enhance import normalize, denormalize

from torchvision.utils import make_grid
from tqdm import tqdm

from data.datasets import CoupledDataset
from src.final_experiments.common_utils import load_ResNet_CelebA, load_StyleGan


def parse_args():

    parser = argparse.ArgumentParser('Test top-1 accuracy of Resnet-50 when interpolating images on StyleGan')

    # TODO for public release, replace "default=..." with "required=True"

    parser.add_argument('--data_path', type=str,
                        default='/media/dserez/datasets/celeba_identities/test/',
                        help='directory of validation set')

    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--resnet_50_path', type=str,
                        default='/media/dserez/runs/adversarial/CNNs/resnet50_celebaa_id_finetuneIN/best.pt',
                        help='directory of cnn to test')

    parser.add_argument('--autoencoder_path', type=str,
                        default='/media/dserez/runs/stylegan2/inversions/e4e_ffhq_encoder.pt',
                        help='checkpoint file for StyleGan+E4E')

    parser.add_argument('--name', type=str, default='test',
                        help='name of experiments for saving results')

    parser.add_argument('--visual_examples_saving_folder', type=str, default='../plots/',
                        help='where to save visual examples of the reconstructed interpolations')

    parser.add_argument('--pickle_file_saving_folder', type=str, default='../results/',
                        help='where to save pickle file with the accuracies')

    args = parser.parse_args()

    # create dirs if not existing
    if not os.path.exists(f'{args.visual_examples_saving_folder}/{args.name}'):
        os.makedirs(f'{args.visual_examples_saving_folder}/{args.name}')

    if not os.path.exists(f'{args.pickle_file_saving_folder}/{args.name}'):
        os.makedirs(f'{args.pickle_file_saving_folder}/{args.name}')

    return args


@torch.no_grad()
def main(data_path: str, resnet_50_path: str, stylegan_path: str,
         pickle_dir: str, plots_dir: str, batch_size: int, device: str = 'cuda:0'):

    # Load resnet
    resnet = load_ResNet_CelebA(resnet_50_path, device)

    # Load StyleNetWork
    autoencoder = load_StyleGan(stylegan_path, device)

    # load dataset
    dataloader = DataLoader(CoupledDataset(folder=data_path, image_size=256,
                                           seed=0), batch_size=batch_size, shuffle=False)

    # structure = {i : {a: accuracy, ...} , ... } where i = latent_idx and a = alpha
    dict_accuracies = {}
    alphas = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], device=device)
    latents_to_test = ['all',] + [i for i in range(18)]

    n_samples = 10
    example_images = torch.zeros((n_samples, len(latents_to_test), len(alphas), 3, 256, 256), device='cpu')

    print('[INFO] Starting Computation on each batch...')

    for batch_idx, batch in enumerate(tqdm(dataloader)):

        # if batch_idx == 2:
        #     break

        # first images and labels, to_interpolate
        x1, y1, x2, y2 = [i.to(device) for i in batch]

        # reconstruct test set
        b, c, h, w = x1.shape

        x = normalize(torch.cat([x1, x2]),
                      torch.tensor([0.5, 0.5, 0.5], device=device),
                      torch.tensor([0.5, 0.5, 0.5], device=device))

        latents = autoencoder.encode(x)
        chunks_x1, chunks_x2 = torch.split(latents, b)

        # interpolate at different alpha terms and check when class is changing
        interpolated_codes = chunks_x1.unsqueeze(1).repeat(1, len(alphas), 1, 1)
        interpolation = chunks_x2.unsqueeze(1).repeat(1, len(alphas), 1, 1)
        alphas = alphas.view(1, -1, 1, 1)
        interpolated_codes = (1 - alphas) * interpolated_codes + alphas * interpolation

        for latent_idx, n in enumerate(latents_to_test):

            if batch_idx == 0:
                dict_accuracies[f'{n}'] = {}

            for alpha_idx, i in enumerate(range(alphas.shape[1])):

                # get recons
                if n == 'all':
                    recons = autoencoder.decode(interpolated_codes[:, i])
                else:
                    initial_codes = interpolated_codes[:, 0].clone()  # alpha 0 == no_interpolation
                    initial_codes[:, n] = interpolated_codes[:, i, n].clone()
                    recons = autoencoder.decode(initial_codes)

                # no need to denormalize generation since predicting!
                preds = torch.argmax(resnet(recons), dim=1)

                a = alphas[0, i].item()

                # append preds to res
                if batch_idx == 0:
                    dict_accuracies[f'{n}'][f'{a:.2f}'] = {
                        'preds': preds.cpu(),
                        'targets': y1.cpu()
                    }
                else:
                    dict_accuracies[f'{n}'][f'{a:.2f}']['preds'] = torch.cat(
                        [dict_accuracies[f'{n}'][f'{a:.2f}']['preds'], preds.cpu()]
                    )
                    dict_accuracies[f'{n}'][f'{a:.2f}']['targets'] = torch.cat(
                        [dict_accuracies[f'{n}'][f'{a:.2f}']['targets'], y1.cpu()]
                    )

                # save picture
                if batch_idx < n_samples:
                    example_images[batch_idx, latent_idx, alpha_idx] = denormalize(recons, 0.5, 0.5)[0].cpu()

    # save pickle
    for k in dict_accuracies.keys():

        all_alphas = dict_accuracies[k]

        for a in all_alphas.keys():
            res = (all_alphas[a]['preds'] == all_alphas[a]['targets']).to(torch.float32).mean()
            del dict_accuracies[k][a]
            dict_accuracies[k][a] = res

    with open(f'{pickle_dir}/class_change_accuracies.pickle', 'wb') as f:
        pickle.dump(dict_accuracies, f)

    # save pictures
    for i, sample in enumerate(example_images):

        display = make_grid(rearrange(sample, 'l a c h w -> (l a) c h w'), nrow=alphas.shape[1])
        display = rearrange(display, 'c h w -> h w c').numpy()
        fig, ax = plt.subplots(figsize=(16, 24))
        ax.imshow(display, interpolation='nearest')
        ax.axis(False)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/chunks_interpolation_sample_{i}.png')
        plt.close()


if __name__ == '__main__':

    arguments = parse_args()

    main(
        data_path=arguments.data_path,
        resnet_50_path=arguments.resnet_50_path,
        stylegan_path=arguments.autoencoder_path,
        plots_dir=f'{arguments.visual_examples_saving_folder}/{arguments.name}',
        pickle_dir=f'{arguments.pickle_file_saving_folder}/{arguments.name}',
        batch_size=arguments.batch_size
    )
