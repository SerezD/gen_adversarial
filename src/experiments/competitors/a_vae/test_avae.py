import argparse
import json

import foolbox as fb
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from tqdm import tqdm

from data.datasets import ImageLabelDataset
from src.defenses.common_utils import load_ResNet_CelebA, load_hub_CNN
from src.experiments.competitors.a_vae.model import StyledGenerator

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class DefenseModel(nn.Module):

    def __init__(self, base_classifier, purifier, kernel_size, classifier_preprocessing = None):
        super().__init__()

        self.base_classifier = base_classifier
        self.purifier = purifier
        self.kernel_size = kernel_size
        self.classifier_preprocessing = classifier_preprocessing

    def forward(self, x):
        """
        x normalization should be done before attack
        """
        x = nn.functional.avg_pool2d(x, self.kernel_size)
        x_cln = self.purifier(x)[2]
        x_cln = (x_cln + 1) / 2

        # from torchvision.utils import make_grid
        # from matplotlib import pyplot as plt
        # display = make_grid(x_cln, nrow=2).permute(1, 2, 0).cpu().numpy()
        # plt.imshow(display)
        # plt.show()

        if self.classifier_preprocessing is not None:
            x_cln = self.classifier_preprocessing(x_cln)

        preds = self.base_classifier(x_cln)
        return preds


def parse_args():

    parser = argparse.ArgumentParser('Test nd_vae defense model')

    parser.add_argument('--images_path', type=str, required=True,
                        help='All images in this folder will be attacked')

    parser.add_argument('--batch_size', type=int, required=True)

    parser.add_argument('--classifier_path', type=str, required=True,
                        help='path to the pre-trained classifier to be attacked')

    parser.add_argument('--classifier_type', type=str, choices=['resnet-32', 'vgg-16', 'resnet-50'],
                        help='type of classifier')

    parser.add_argument('--autoencoder_path', type=str, required=True,
                        help='path to the pre-trained autoencoder acting as a defense')

    parser.add_argument('--results_folder', type=str, required=True,
                        help='folder to save .pickle file with results')

    args = parser.parse_args()

    # create folder
    args.results_folder = f'{args.results_folder}/{args.classifier_type}/'
    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    return args


def main(args):

    # load pre-trained models
    if args.classifier_type == 'resnet-32':
        base_classifier = load_hub_CNN(args.classifier_path, args.classifier_type, device)
        KERNEL_SIZE = 1  # TODO
        a_vae = None  # TODO
        classifier_preprocessing = None
        args.image_size = 32
        bounds_l2 = (0.5, 1.0, 2.0, 4.0)
    elif args.classifier_type == 'vgg-16':
        base_classifier = load_hub_CNN(args.classifier_path, args.classifier_type, device)
        KERNEL_SIZE = 1  # TODO
        a_vae = None  # TODO
        classifier_preprocessing = None
        args.image_size = 32
        bounds_l2 = (0.5, 1.0, 2.0, 4.0)
    elif args.classifier_type == 'resnet-50':
        base_classifier = load_ResNet_CelebA(args.classifier_path, device)

        KERNEL_SIZE = 8
        a_vae = StyledGenerator(512, 256 // KERNEL_SIZE, 256)

        state_dict = torch.load(args.autoencoder_path)
        a_vae.load_state_dict(state_dict)
        a_vae.to(device).eval()

        classifier_preprocessing = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        args.image_size = 256
        bounds_l2 = (0.5, 1.0, 2.0, 4.0)  # TODO CHECK BOUNDS
    else:
        raise ValueError(f'Unknown classifier type: {args.classifier_type}')

    defense_model = DefenseModel(base_classifier, a_vae, KERNEL_SIZE, classifier_preprocessing).eval()

    # wrap defense with EOT to maintain deterministic predictions
    # this is a simple counter_attack, to avoid gradient masking
    fb_defense_model = fb.PyTorchModel(defense_model, bounds=(0, 1), device=device,
                                       preprocessing=dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], axis=-3))
    fb_defense_model = fb.models.ExpectationOverTransformationWrapper(fb_defense_model, n_steps=8)

    # dataloader
    dataloader = DataLoader(ImageLabelDataset(folder=args.images_path, image_size=args.image_size),
                            batch_size=args.batch_size, shuffle=False)

    # attacks
    n_steps = 15  # reduce number of tests
    attacks = [
        fb.attacks.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack(repeats=96),
        fb.attacks.L2DeepFoolAttack(steps=50,  candidates=10, overshoot=0.02),
        # fb.attacks.L2CarliniWagnerAttack(binary_search_steps=4, steps=2048)

        # TODO TEST
        fb.attacks.L2CarliniWagnerAttack(binary_search_steps=2,
                                         steps=4096,
                                         stepsize=5e-3,
                                         confidence=1e-2,
                                         initial_const=1)

        # fb.attacks.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack(repeats=4),
        # fb.attacks.L2DeepFoolAttack(steps=50, candidates=10, overshoot=0.02),
        # fb.attacks.L2CarliniWagnerAttack(binary_search_steps=1, steps=4)

    ]

    success_rates = torch.empty((3, len(bounds_l2), 0), device=device)

    for idx, (images, labels) in enumerate(tqdm(dataloader)):

        # if idx > 20:
        #     break

        if idx % n_steps != 0:
            continue

        images = torch.clip(images.to(device), 0.0, 1.0)
        labels = labels.to(device)

        # uncomment to visualize some examples
        # from torchvision.utils import make_grid
        # from matplotlib import pyplot as plt
        # x_cln = a_vae(nn.functional.avg_pool2d(((images * 2) - 1), KERNEL_SIZE))[2]
        # x_cln = (x_cln + 1) / 2
        # display = make_grid(torch.cat((images, x_cln), dim=0), nrow=args.batch_size).permute(1, 2, 0).cpu().numpy()
        # plt.imshow(display)
        # plt.show()
        # continue

        # TODO measure clean

        # TODO Remove refactoring for C&W
        # # BRUTE FORCE
        # with torch.no_grad():
        #     success_0 = attacks[0](fb_defense_model, images, labels, epsilons=bounds_l2)[2]
        #
        # # DEEP FOOL
        # success_1 = attacks[1](fb_defense_model, images, labels, epsilons=bounds_l2)[2]

        # C&W
        success_2 = attacks[2](fb_defense_model, images, labels, epsilons=bounds_l2)[2]

        # success_rates = torch.cat((success_rates,
        #                             torch.stack((success_0, success_1, success_2), dim=0)), dim=2)

        success_rates = torch.cat((success_rates,
                                    torch.stack((success_2, success_2, success_2), dim=0)), dim=2)

    res_dict = {}
    for a_i, attack in enumerate(['BruteForce', 'DeepFool', 'C&W']):
        for b_i, bound in enumerate(bounds_l2):
            res_dict[f'{attack}_{bound:.2f}'] = success_rates[a_i][b_i].mean().item()

    # TODO REMOVE NAME REFACTOR
    with open(f'{args.results_folder}results_C&W.json', 'w') as f:
        json.dump(res_dict, f)


if __name__ == '__main__':

    main(parse_args())