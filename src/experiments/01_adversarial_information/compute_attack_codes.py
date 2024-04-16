import argparse

import numpy as np
from einops import pack
import foolbox as fb
from kornia.augmentation import Normalize
import os
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import ImageLabelDataset
from src.defenses.common_utils import load_StyleGan, load_NVAE, load_ResNet_CelebA, load_hub_CNN

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def parse_args():
    """
    CONFS

    --images_folder /media/dserez/datasets/celeba_hq_gender/validation/
    --batch_size 10
    --autoencoder_path /media/dserez/runs/stylegan2/inversions/e4e_ffhq_encoder.pt
    --autoencoder_name StyleGan_E4E
    --classifier_path /media/dserez/runs/adversarial/CNNs/resnet50_celeba_hq_gender/best.pt
    --classifier_type resnet-50
    --attack DeepFool
    --bound_magnitude 0.25
    --results_folder /media/dserez/runs/adversarial/01_adversarial_information/pre_computed_codes/

    --images_folder /media/dserez/datasets/celeba_hq_gender/
    --batch_size 10
    --autoencoder_path /media/dserez/runs/stylegan2/inversions/e4e_ffhq_encoder.pt
    --autoencoder_name StyleGan_E4E
    --classifier_path /media/dserez/runs/adversarial/CNNs/resnet50_celeba_hq_gender/best.pt
    --classifier_type resnet-50
    --attack DeepFool
    --bound_magnitude 0.50
    --results_folder /media/dserez/runs/adversarial/01_adversarial_information/pre_computed_codes/

    --images_folder /media/dserez/datasets/celeba_hq_gender/
    --batch_size 10
    --autoencoder_path /media/dserez/runs/stylegan2/inversions/e4e_ffhq_encoder.pt
    --autoencoder_name StyleGan_E4E
    --classifier_path /media/dserez/runs/adversarial/CNNs/resnet50_celeba_hq_gender/best.pt
    --classifier_type resnet-50
    --attack DeepFool
    --bound_magnitude 1.00
    --results_folder /media/dserez/runs/adversarial/01_adversarial_information/pre_computed_codes/

    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    --images_folder /media/dserez/datasets/celeba_hq_gender/
    --batch_size 10
    --autoencoder_path /media/dserez/runs/stylegan2/inversions/e4e_ffhq_encoder.pt
    --autoencoder_name StyleGan_E4E
    --classifier_path /media/dserez/runs/adversarial/CNNs/resnet50_celeba_hq_gender/best.pt
    --classifier_type resnet-50
    --attack C&W
    --bound_magnitude 0.25
    --results_folder /media/dserez/runs/adversarial/01_adversarial_information/pre_computed_codes/

    --images_folder /media/dserez/datasets/celeba_hq_gender/
    --batch_size 10
    --autoencoder_path /media/dserez/runs/stylegan2/inversions/e4e_ffhq_encoder.pt
    --autoencoder_name StyleGan_E4E
    --classifier_path /media/dserez/runs/adversarial/CNNs/resnet50_celeba_hq_gender/best.pt
    --classifier_type resnet-50
    --attack C&W
    --bound_magnitude 0.50
    --results_folder /media/dserez/runs/adversarial/01_adversarial_information/pre_computed_codes/

    --images_folder /media/dserez/datasets/celeba_hq_gender/
    --batch_size 10
    --autoencoder_path /media/dserez/runs/stylegan2/inversions/e4e_ffhq_encoder.pt
    --autoencoder_name StyleGan_E4E
    --classifier_path /media/dserez/runs/adversarial/CNNs/resnet50_celeba_hq_gender/best.pt
    --classifier_type resnet-50
    --attack C&W
    --bound_magnitude 1.00
    --results_folder /media/dserez/runs/adversarial/01_adversarial_information/pre_computed_codes/

    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
    parser = argparse.ArgumentParser('Compute latent codes given HL autoencoder '
                                     'for attacked val set of given classifier.')

    parser.add_argument('--images_folder', type=str, required=True,
                        help='Folder with validation set images')

    parser.add_argument('--batch_size', type=int, required=True)

    parser.add_argument('--autoencoder_path', type=str, required=True,
                        help='path to the pre-trained hl autoencoder used to get codes')

    parser.add_argument('--autoencoder_name', type=str, required=True,
                        help='used to determine results folder, must contain substrings NVAE or StyleGAN')

    parser.add_argument('--results_folder', type=str, required=True,
                        help='folder to save .pickle file with results')

    parser.add_argument('--classifier_path', type=str, required=True,
                        help='path to the pre-trained classifier to be attacked')

    parser.add_argument('--classifier_type', type=str, choices=['resnet-32', 'vgg-16', 'resnet-50'],
                        help='type of classifier')

    parser.add_argument('--attack', type=str, choices=['BruteForce', 'DeepFool', 'C&W'])
    parser.add_argument('--bound_magnitude', type=float, help='L2 bound magnitude for attack')

    args = parser.parse_args()

    assert 'nvae' in args.autoencoder_name.lower() or 'stylegan' in args.autoencoder_name.lower(), \
        'argument --autoencoder_name: used to determine results folder, must contain substrings NVAE or StyleGAN'

    # create folder
    args.results_folder = f'{args.results_folder}/{args.autoencoder_name}/{args.classifier_type}/'
    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    return args


def main(args: argparse.Namespace):

    # load pretrained cnn
    if args.classifier_type == 'resnet-50':
        classifier = load_ResNet_CelebA(args.classifier_path, device)
        preprocessing_cl = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], axis=-3)

    else:
        classifier = load_hub_CNN(args.classifier_path, args.classifier_type, device)
        preprocessing_cl= dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)

    # load pre-trained autoencoder
    if 'nvae' in args.autoencoder_name.lower():
        hl_autoencoder = load_NVAE(args.autoencoder_path, device)
        preprocessing_ae = None
        args.image_size = 32
    else:
        hl_autoencoder = load_StyleGan(args.autoencoder_path, device)
        preprocessing_ae = Normalize(mean=torch.tensor([0.5, 0.5, 0.5], device=device),
                                  std=torch.tensor([0.5, 0.5, 0.5], device=device))
        args.image_size = 256

    # dataloader
    valid_dataloader = DataLoader(ImageLabelDataset(folder=args.images_folder, image_size=args.image_size),
                                  batch_size=args.batch_size, shuffle=False)

    # attack info
    attacked_model = fb.PyTorchModel(classifier, bounds=(0, 1), preprocessing=preprocessing_cl, device=device)

    if args.attack == 'C&W':
        # default requires unrealistic amount of time!
        attack = fb.attacks.L2CarliniWagnerAttack(binary_search_steps=1, steps=2048)
    elif args.attack == 'DeepFool':
        attack = fb.attacks.L2DeepFoolAttack()
    else:
        attack = fb.attacks.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack(repeats=128)

    print('[INFO] Computing codes on validation set...')
    valid_codes = {}
    success_rates = torch.empty((0, ), dtype=torch.bool, device=device)  # boolean tensor of len(dataset)

    for images, labels in tqdm(valid_dataloader):

        images, labels = torch.clamp(images.to(device), 0., 1.), labels.to(device)

        # attacked_batch
        _, adversaries, success_rate = attack(attacked_model, images, labels, epsilons=args.bound_magnitude)
        success_rates = torch.cat((success_rates, success_rate), dim=0)

        with torch.no_grad():
            if preprocessing_ae is not None:
                adversaries = preprocessing_ae(adversaries)

            # list of codes - each of shape BS, n
            codes = hl_autoencoder.get_codes(adversaries)

            for i, c in enumerate(codes):
                if i not in valid_codes:
                    valid_codes[i] = c.cpu().numpy()
                else:
                    valid_codes[i] = pack([valid_codes[i], c.cpu().numpy()], '* n')[0]

    file_name = f'{args.results_folder}/{args.attack}_codes_l2={args.bound_magnitude}.pickle'
    with open(file_name, 'wb') as f:
        pickle.dump(valid_codes, f)
    print(f'[INFO] saved to {file_name}')

    file_name = f'{args.results_folder}/{args.attack}_codes_l2={args.bound_magnitude}_success_rates.npy'
    np.save(file_name, success_rates.cpu().numpy())
    print(f'[INFO] saved to {file_name}')


if __name__ == '__main__':
    """
    Compute latent codes after attack on validation set.
    """
    arguments = parse_args()
    main(arguments)
