import argparse
from einops import pack
from kornia.augmentation import Normalize
import os
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import ImageDataset
from src.defenses.common_utils import load_StyleGan, load_NVAE


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def parse_args():
    """
    CONFS

    --images_folder
    /media/dserez/datasets/celeba_hq_gender/
    --batch_size
    20
    --autoencoder_path
    /media/dserez/runs/stylegan2/inversions/e4e_ffhq_encoder.pt
    --autoencoder_name
    StyleGan_E4E
    --results_folder
    /media/dserez/runs/adversarial/01_adversarial_information/pre_computed_codes/
    --bound_magnitude 2.0

    --images_folder
    /media/dserez/datasets/cifar10/
    --batch_size
    50
    --autoencoder_path
    /media/dserez/runs/NVAE/cifar10/ours/3scales_4groups.pt
    --autoencoder_name
    NVAE3x4
    --results_folder
    /media/dserez/runs/adversarial/01_adversarial_information/pre_computed_codes/

    --images_folder
    /media/dserez/datasets/cifar10/
    --batch_size
    50
    --autoencoder_path
    /media/dserez/runs/NVAE/cifar10/ours/replica/large_latents.pt
    --autoencoder_name
    NVAE3x1
    --results_folder
    /media/dserez/runs/adversarial/01_adversarial_information/pre_computed_codes/

    """
    parser = argparse.ArgumentParser('Compute latent codes given HL autoencoder for full train/val sets.')

    parser.add_argument('--images_folder', type=str, required=True,
                        help='Folder with train and validation subfolders')

    parser.add_argument('--batch_size', type=int, required=True)

    parser.add_argument('--autoencoder_path', type=str, required=True,
                        help='path to the pre-trained hl autoencoder used to get codes')

    parser.add_argument('--autoencoder_name', type=str, required=True,
                        help='used to determine results folder, must contain substrings NVAE or StyleGAN')

    parser.add_argument('--results_folder', type=str, required=True,
                        help='folder to save .pickle file with results')

    parser.add_argument('--bound_magnitude', type=float, default=0.0,
                        help='L2 bound magnitude for noise')

    args = parser.parse_args()

    assert 'nvae' in args.autoencoder_name.lower() or 'stylegan' in args.autoencoder_name.lower(), \
        'argument --autoencoder_name: used to determine results folder, must contain substrings NVAE or StyleGAN'

    # create folder
    args.results_folder = f'{args.results_folder}/{args.autoencoder_name}/'
    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    return args


def get_noisy_batch(batch: torch.Tensor, eps: float, clip_min: float = 0.0, clip_max: float = 1.0) -> torch.Tensor:
    """
    returns noisy batch within L2 bound = eps.
    If eps is zero, returns batch.
    """

    # get random noise and its l2 norm
    noise = torch.zeros_like(batch).uniform_(-clip_max / 2, clip_max / 2)

    l2_norm_noise = torch.max(torch.ones_like(noise) * 1e-12, torch.sum(noise ** 2, dim=[1, 2, 3], keepdim=True))
    l2_norm_noise = torch.sqrt(l2_norm_noise)

    # get the normalized noise
    mul_factor = torch.min(torch.ones_like(noise), eps / l2_norm_noise)
    noise = noise * mul_factor

    # clamp in range and return
    batch = torch.clamp(batch + noise, clip_min, clip_max)

    return batch


@torch.no_grad()
def main(args: argparse.Namespace):

    # load pre-trained autoencoder
    if 'nvae' in args.autoencoder_name.lower():
        hl_autoencoder = load_NVAE(args.autoencoder_path, device)
        preprocessing = None
        args.image_size = 32
    else:
        hl_autoencoder = load_StyleGan(args.autoencoder_path, device)
        preprocessing = Normalize(mean=torch.tensor([0.5, 0.5, 0.5], device=device),
                                  std=torch.tensor([0.5, 0.5, 0.5], device=device))
        args.image_size = 256

    # dataloader
    train_dataloader = DataLoader(ImageDataset(folder=f'{args.images_folder}/train/', image_size=args.image_size),
                                  batch_size=args.batch_size, shuffle=False)
    valid_dataloader = DataLoader(ImageDataset(folder=f'{args.images_folder}/validation/', image_size=args.image_size),
                                  batch_size=args.batch_size, shuffle=False)

    print('[INFO] Computing codes on training set...')
    train_codes = {}

    for batch in tqdm(train_dataloader):

        batch = batch.to(device)

        # add noise
        batch = get_noisy_batch(batch, args.bound_magnitude)

        if preprocessing is not None:
            batch = preprocessing(batch)

        # list of codes - each of shape BS, n
        codes = hl_autoencoder.get_codes(batch)

        for i, c in enumerate(codes):
            if i not in train_codes:
                train_codes[i] = c.cpu().numpy()
            else:
                train_codes[i] = pack([train_codes[i], c.cpu().numpy()], '* n')[0]

    file_name = f'{args.results_folder}/train_codes'
    if args.bound_magnitude > 0:
        file_name += f'_l2={args.bound_magnitude}'
    file_name += '.pickle'

    with open(file_name, 'wb') as f:
        pickle.dump(train_codes, f)
    print(f'[INFO] saved to {file_name}')

    print('[INFO] Computing codes on validation set...')
    valid_codes = {}

    for batch in tqdm(valid_dataloader):

        batch = batch.to(device)

        # add noise
        batch = get_noisy_batch(batch, args.bound_magnitude)

        if preprocessing is not None:
            batch = preprocessing(batch)

        # list of codes - each of shape BS, n
        codes = hl_autoencoder.get_codes(batch)

        for i, c in enumerate(codes):
            if i not in valid_codes:
                valid_codes[i] = c.cpu().numpy()
            else:
                valid_codes[i] = pack([valid_codes[i], c.cpu().numpy()], '* n')[0]

    file_name = f'{args.results_folder}/valid_codes'
    if args.bound_magnitude > 0:
        file_name += f'_l2={args.bound_magnitude}'
    file_name += '.pickle'

    with open(file_name, 'wb') as f:
        pickle.dump(valid_codes, f)
    print(f'[INFO] saved to {file_name}')


if __name__ == '__main__':
    """
    Compute latent codes given HL autoencoder for full train/val sets.
    """
    arguments = parse_args()
    main(arguments)
