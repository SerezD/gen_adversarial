import argparse

import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import ImageNameLabelDataset
from src.attacks.untargeted import FGSM
from src.defenses.ours.models import CelebaGenderClassifier, E4EStyleGanDefenseModel, CelebaIdentityClassifier, \
    NVAEDefenseModel, CarsTypeClassifier, TransStyleGanDefenseModel
from src.defenses.wrappers import EoTWrapper

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def parse_args():
    """

    """
    parser = argparse.ArgumentParser('')

    parser.add_argument('--images_folder', type=str, required=True,
                        help='Folder with images to make adversarial')

    parser.add_argument('--n_samples', type=int, required=True,
                        help='You likely want to run BO on a subset of the whole training dataset.')

    parser.add_argument('--results_folder', type=str, required=True,
                        help='folder to save image adversaries')

    parser.add_argument('--classifier_path', type=str, required=True,
                        help='path to the pre-trained classifier to be attacked')

    parser.add_argument('--autoencoder_path', type=str, required=True,
                        help='path to the pre-trained HL Autoencoder')

    parser.add_argument('--classifier_type', type=str, choices=['resnet-50', 'vgg-11', 'resnext-50'],
                        help='type of classifier')

    args = parser.parse_args()

    return args


def main(args: argparse.Namespace):

    # load and initiate stuff
    if args.classifier_type == 'resnet-50':
        args.image_size = 256

        base_classifier = CelebaGenderClassifier(args.classifier_path, device)

        interpolation_alphas = [0. for _ in range(18)]  # reconstruction only
        defense_model = E4EStyleGanDefenseModel(base_classifier, args.autoencoder_path,
                                                interpolation_alphas, device=device).eval()
        attack = FGSM(l2_bound=4.)

    elif args.classifier_type == 'vgg-11':
        args.image_size = 64

        base_classifier = CelebaIdentityClassifier(args.classifier_path, device)

        interpolation_alphas = [0. for _ in range(24)]  # reconstruction only
        defense_model = NVAEDefenseModel(base_classifier, args.autoencoder_path,
                                         interpolation_alphas, device=device).eval()

        attack = FGSM(l2_bound=2.)

    elif args.classifier_type == 'resnext-50':
        args.image_size = 128

        base_classifier = CarsTypeClassifier(args.classifier_path, device)

        interpolation_alphas = [0. for _ in range(16)]  # reconstruction only
        defense_model = TransStyleGanDefenseModel(base_classifier, args.autoencoder_path,
                                                  interpolation_alphas, device=device).eval()

        attack = FGSM(l2_bound=4.)

    else:
        raise NotImplemented

    # Probably not needed, since we are doing reconstruction only.
    # However, some non-deterministic stuff may be part of the standard forward pass of generator.
    defense_model = EoTWrapper(defense_model, eot_steps=32).eval()

    # dataloaders
    train_dataloader = DataLoader(ImageNameLabelDataset(folder=args.images_folder, image_size=args.image_size),
                                  batch_size=1, shuffle=True)  # -> shuffle to allow samples from all classes

    found_samples = 0
    for (image, name, label) in tqdm(train_dataloader):

        if found_samples == args.n_samples:
            break

        image, label = torch.clamp(image.to(device), 0., 1.), label.to(device)

        # adversaries
        success, bound, adversary = attack(image, label, defense_model)

        if success and bound > 0.:

            found_samples += 1

            # save
            folder_name = f'{args.results_folder}/{name[0][0]}/'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            file_name = f'{folder_name}/{name[1][0]}'
            Image.fromarray((adversary[0] * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)).save(file_name)


if __name__ == '__main__':
    """
    Pre-Generate an adversarial version of the dataset to optimize alpha values.
    """

    arguments = parse_args()
    main(arguments)
