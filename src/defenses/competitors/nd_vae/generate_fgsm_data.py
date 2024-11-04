import argparse

import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import ImageNameLabelDataset
from src.attacks.untargeted import FGSM
from src.defenses.ours.models import CelebaGenderClassifier, CelebaIdentityClassifier, CarsTypeClassifier

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def parse_args():
    """

    """
    parser = argparse.ArgumentParser('')

    parser.add_argument('--images_folder', type=str, required=True,
                        help='Folder with training set images')

    parser.add_argument('--results_folder', type=str, required=True,
                        help='folder to save adversarial results')

    parser.add_argument('--classifier_path', type=str, required=True,
                        help='path to the pre-trained classifier to be attacked')

    parser.add_argument('--classifier_type', type=str, choices=['resnet-50', 'vgg-11', 'resnext-50'],
                        help='type of classifier')

    args = parser.parse_args()

    return args


def main(args: argparse.Namespace):

    # load pretrained cnn
    if args.classifier_type == 'resnet-50':
        classifier = CelebaGenderClassifier(args.classifier_path, device)
        attack = FGSM(l2_bound=4.)
        args.image_size = 256

    elif args.classifier_type == 'vgg-11':
        classifier = CelebaIdentityClassifier(args.classifier_path, device)
        attack = FGSM(l2_bound=2.)
        args.image_size = 64

    elif args.classifier_type == 'resnext-50':
        classifier = CarsTypeClassifier(args.classifier_path, device)
        attack = FGSM(l2_bound=4.)
        args.image_size = 128

    else:
        raise ValueError('Invalid classifier type')

    # dataloaders
    train_dataloader = DataLoader(ImageNameLabelDataset(folder=f'{args.images_folder}', image_size=args.image_size),
                                  batch_size=1, shuffle=False)

    for b_idx, (image, name, label) in enumerate(tqdm(train_dataloader)):

        image, label = torch.clamp(image.to(device), 0., 1.), label.to(device)

        # adversaries
        _, _, adversary = attack(image, label, classifier)

        # save
        folder_name = f'{args.results_folder}/{name[0][0]}/'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        file_name = f'{folder_name}/{name[1][0]}'
        Image.fromarray((adversary[0] * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)).save(file_name)


if __name__ == '__main__':
    """
    ND-VAE uses adversarial data for training.
    Pre-Generate an adversarial version of the dataset.
    """

    arguments = parse_args()
    main(arguments)
