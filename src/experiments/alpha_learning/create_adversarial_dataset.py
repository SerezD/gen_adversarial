import argparse

import numpy as np
from PIL import Image
import foolbox as fb
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import ImageNameLabelDataset
from src.defenses.models import CelebAResnetModel, CelebAStyleGanDefenseModel

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def parse_args():
    """

    """
    parser = argparse.ArgumentParser('')

    parser.add_argument('--images_folder', type=str, required=True,
                        help='Folder with train/validation images')

    parser.add_argument('--batch_size', type=int, required=True)

    parser.add_argument('--results_folder', type=str, required=True,
                        help='folder to save image adversaries')

    parser.add_argument('--classifier_path', type=str, required=True,
                        help='path to the pre-trained classifier to be attacked')

    parser.add_argument('--autoencoder_path', type=str, required=True,
                        help='path to the pre-trained HL Autoencoder')

    parser.add_argument('--classifier_type', type=str, choices=['resnet-50', 'vgg-16'],
                        help='type of classifier')

    args = parser.parse_args()

    return args


def main(args: argparse.Namespace):

    # load pretrained purification model
    if args.classifier_type == 'resnet-50':
        args.image_size = 256
        args.bound = (4., )
        args.interpolation_alphas = [0. for _ in range(18)]  # reconstruction only
        args.initial_noise_eps = 0.
        args.gaussian_blur_input = False
        base_classifier = CelebAResnetModel(args.classifier_path, device)
        defense_model = CelebAStyleGanDefenseModel(base_classifier, args.autoencoder_path,
                                                   args.interpolation_alphas, args.initial_noise_eps,
                                                   args.gaussian_blur_input, device)

    else:
        raise NotImplemented

    # dataloaders
    train_dataloader = DataLoader(ImageNameLabelDataset(folder=f'{args.images_folder}/train/',
                                                        image_size=args.image_size),
                                  batch_size=args.batch_size, shuffle=False)

    valid_dataloader = DataLoader(ImageNameLabelDataset(folder=f'{args.images_folder}/validation/',
                                                        image_size=args.image_size),
                                  batch_size=args.batch_size, shuffle=False)

    # attack info
    attacked_model = fb.PyTorchModel(defense_model, bounds=(0, 1), device=device)

    attack = fb.attacks.L2FastGradientAttack()

    for b_idx, (images, names, labels) in enumerate(tqdm(train_dataloader)):

        images, labels = torch.clamp(images.to(device), 0., 1.), labels.to(device)

        # attacked_batch
        _, adversaries, _ = attack(attacked_model, images, labels, epsilons=args.bound)

        # save
        for i, img in enumerate(adversaries[0]):

            folder_name = f'{args.results_folder}/train/{labels[i].item()}/'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            file_name = f'{folder_name}/{names[i]}'
            Image.fromarray((img * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)).save(file_name)

    for b_idx, (images, names, labels) in enumerate(tqdm(valid_dataloader)):

        images, labels = torch.clamp(images.to(device), 0., 1.), labels.to(device)

        # attacked_batch
        _, adversaries, _ = attack(attacked_model, images, labels, epsilons=args.bound)

        # save
        for i, img in enumerate(adversaries[0]):

            folder_name = f'{args.results_folder}/validation/{labels[i].item()}/'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            file_name = f'{folder_name}/{names[i]}'
            Image.fromarray((img * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)).save(file_name)


if __name__ == '__main__':
    """
    Pre-Generate an adversarial version of the dataset to optimize alpha values.
    """

    arguments = parse_args()
    main(arguments)
