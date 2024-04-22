import argparse

import numpy as np
from PIL import Image
import foolbox as fb
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import ImageLabelDataset, ImageNameLabelDataset
from src.defenses.common_utils import load_ResNet_CelebA, load_hub_CNN

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def parse_args():
    """

    """
    parser = argparse.ArgumentParser('')

    parser.add_argument('--images_folder', type=str, required=True,
                        help='Folder with validation set images')

    parser.add_argument('--batch_size', type=int, required=True)

    parser.add_argument('--results_folder', type=str, required=True,
                        help='folder to save .pickle file with results')

    parser.add_argument('--classifier_path', type=str, required=True,
                        help='path to the pre-trained classifier to be attacked')

    parser.add_argument('--classifier_type', type=str, choices=['resnet-32', 'vgg-16', 'resnet-50'],
                        help='type of classifier')

    args = parser.parse_args()

    return args


def main(args: argparse.Namespace):

    # load pretrained cnn
    if args.classifier_type == 'resnet-50':
        classifier = load_ResNet_CelebA(args.classifier_path, device)
        preprocessing_cl = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], axis=-3)
        args.image_size = 256
        args.bound = (2., )

    else:
        classifier = load_hub_CNN(args.classifier_path, args.classifier_type, device)
        preprocessing_cl= dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
        args.image_size = 32
        args.bound = (0.5,)

    # dataloaders
    train_dataloader = DataLoader(ImageNameLabelDataset(folder=f'{args.images_folder}/train/',
                                                        image_size=args.image_size),
                                  batch_size=args.batch_size, shuffle=False)

    # attack info
    attacked_model = fb.PyTorchModel(classifier, bounds=(0, 1), preprocessing=preprocessing_cl, device=device)

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


if __name__ == '__main__':
    """
    ND-VAE uses adversarial data for training.
    Pre-Generate an adversarial version of the dataset.
    """

    arguments = parse_args()
    main(arguments)
