import argparse
import json
import os
import torch
from einops import pack, rearrange
import foolbox as fb
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from data.datasets import ImageLabelDataset
from src.defenses.models import Cifar10ResnetModel, CelebAResnetModel, Cifar10VGGModel, Cifar10NVAEDefenseModel, \
    CelebAStyleGanDefenseModel


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def parse_args():

    """

    """

    parser = argparse.ArgumentParser('Test the defense model with some decision based attacks')

    parser.add_argument('--images_path', type=str, required=True,
                        help='All images in this folder will be attacked')

    parser.add_argument('--batch_size', type=int, required=True)

    parser.add_argument('--classifier_path', type=str, required=True,
                        help='path to the pre-trained classifier to be attacked')

    parser.add_argument('--classifier_type', type=str, choices=['resnet-32', 'vgg-16', 'resnet-50'],
                        help='type of classifier')

    parser.add_argument('--autoencoder_path', type=str, required=True,
                        help='path to the pre-trained autoencoder acting as a defense')

    parser.add_argument('--autoencoder_name', type=str, required=True,
                        help='used to determine results folder')

    parser.add_argument('--resample_from', type=int, required=True,
                        help='hierarchy level where re-sampling defense starts')

    parser.add_argument('--results_folder', type=str, required=True,
                        help='folder to save .pickle file with results')

    args = parser.parse_args()

    # create folder
    args.results_folder = f'{args.results_folder}/{args.autoencoder_name}_{args.classifier_type}/'
    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    args.plots_folder = f'{args.results_folder}/plots_brute_force/'
    if not os.path.exists(args.plots_folder):
        os.makedirs(args.plots_folder)

    return args


@torch.no_grad()
def main(args: argparse.Namespace):
    """
    Try FoolBox Implemented Decision-Based Attacks
    - InversionAttack PointWiseAttack BoundaryAttack
    - Can be Targeted ?
    - Plot success rate vs perturbation rate until reaching 100 % success rate.
    """

    # load pre-trained models
    if args.classifier_type == 'resnet-32':
        base_classifier = Cifar10ResnetModel(args.classifier_path, device)
        defense_model = Cifar10NVAEDefenseModel(base_classifier, args.autoencoder_path, device, args.resample_from)
        args.image_size = 32
        bounds_l2 = (0.1, 0.5, 1.0)
    elif args.classifier_type == 'vgg-16':
        base_classifier = Cifar10VGGModel(args.classifier_path, device)
        defense_model = Cifar10NVAEDefenseModel(base_classifier, args.autoencoder_path, device, args.resample_from)
        args.image_size = 32
        bounds_l2 = (0.1, 0.5, 1.0)
    elif args.classifier_type == 'resnet-50':
        base_classifier = CelebAResnetModel(args.classifier_path, device)
        defense_model = CelebAStyleGanDefenseModel(base_classifier, args.autoencoder_path, device, args.resample_from)
        args.image_size = 256
        bounds_l2 = (0.5, 1.0, 2.0)
    else:
        raise ValueError(f'Unknown classifier type: {args.classifier_type}')

    base_model = base_classifier.classifier.clone().eval()
    defense_model = defense_model.eval()

    # dataloader
    dataloader = DataLoader(ImageLabelDataset(folder=args.images_path, image_size=args.image_size),
                            batch_size=args.batch_size, shuffle=False)


    # params attacks
    # TODO Check preprocessing is correct!
    def_attacked_model = fb.PyTorchModel(defense_model, bounds=(0, 1), device=device)

    if base_classifier.preprocessing is not None:
        preprocessing = dict(mean=base_classifier.mean, std=base_classifier.std, axis=-3)
    else:
        preprocessing = None

    base_attacked_model = fb.PyTorchModel(base_model, bounds=(0, 1), device=device,
                                          preprocessing=preprocessing)

    # TODO implement attacks!
    attack = fb.attacks.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack(repeats=100)

    base_success_rate = torch.empty((len(bounds_l2), 0), device=device)
    def_success_rate = torch.empty((len(bounds_l2), 0), device=device)

    for b_idx, (images, labels) in enumerate(tqdm(dataloader)):

        # if b_idx == 2:  # TODO REMOVE
        #     break

        images = torch.clip(images.to(device), 0.0, 1.0)
        labels = labels.to(device)

        # Brute Force Attack with increasing L2 bounds.
        # adv = list of len(bounds_l2) of tensors B C H W.
        # success = boolean tensor of shape (N_bounds, B)
        _, _, suc_base = attack(base_attacked_model, images, labels, epsilons=bounds_l2)
        _, adv, suc_def = attack(def_attacked_model, images, labels, epsilons=bounds_l2)

        # cat results
        base_success_rate, _ = pack([base_success_rate, suc_base], 'n *')
        def_success_rate, _ = pack([def_success_rate, suc_def], 'n *')

        # save visual examples of applied perturbation.
        if b_idx % 20 == 0:

            # estimate of the cleaned image (estimate because the process is random)
            cleaned_imgs = [defense_model(found_adv, preds_only=False)[-1] for found_adv in adv]

            for i, b in enumerate(bounds_l2):

                max_b = 8
                img_i = images.clone()[:max_b]
                adv_i = adv[i][:max_b]
                def_i = cleaned_imgs[i][:max_b]
                max_b = adv_i.shape[0]

                examples = torch.cat([img_i, adv_i, def_i], dim=0)
                display = make_grid(examples, nrow=max_b).permute(1, 2, 0).cpu()
                plt.imshow(display)
                plt.axis(False)
                plt.title(f'originals, adversarial and cleaned images at L2={b:.2f}')
                plt.savefig(f'{args.plots_folder}batch_{b_idx}_bound={b:.2f}.png')
                plt.close()

    # compute and save final results
    res_dict = {}

    for i, b in enumerate(bounds_l2):
        res_dict[f'brute force l2 bound={b:.2f} success base'] = base_success_rate[i].mean().item()
        res_dict[f'brute force l2 bound={b:.2f} success defense'] = def_success_rate[i].mean().item()

    with open(f'{args.results_folder}results.json', 'w') as f:
        json.dump(res_dict, f)


if __name__ == '__main__':

    arguments = parse_args()
    main(arguments)

