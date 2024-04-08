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

from data.datasets import CoupledDataset
from src.defenses.models import Cifar10ResnetModel, CelebAResnetModel, Cifar10VGGModel, Cifar10NVAEDefenseModel, \
    CelebAStyleGanDefenseModel


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def parse_args():

    """
    STYLEGAN
    --images_path
    /media/dserez/datasets/celeba_hq_gender/test/
    --batch_size
    4
    --classifier_path
    /media/dserez/runs/adversarial/CNNs/resnet50_celeba_hq_gender/best.pt
    --classifier_type
    resnet-50
    --autoencoder_path
    /media/dserez/runs/stylegan2/inversions/e4e_ffhq_encoder.pt
    --autoencoder_name
    E4E_StyleGAN
    --resample_from
    9
    --results_folder
    ./tmp/
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

    base_model = base_classifier.classifier.eval()
    defense_model = defense_model.eval()

    # dataloader
    dataloader = DataLoader(CoupledDataset(folder=args.images_path, image_size=args.image_size),
                            batch_size=args.batch_size, shuffle=False)

    # create FoolBox Models (move preprocessing phase before attack)
    if defense_model.preprocess:
        preprocessing = dict(mean=defense_model.mean, std=defense_model.std, axis=-3)
        defense_model.preprocess = False
    else:
        preprocessing = None

    # wrap defense with EOT to maintain deterministic predictions
    # this is a simple counter_attack, otherwise HSJA would fail by default
    fb_defense_model = fb.PyTorchModel(defense_model, bounds=(0, 1), device=device,
                                       preprocessing=preprocessing)
    fb_defense_model = fb.models.ExpectationOverTransformationWrapper(fb_defense_model, n_steps=64)

    if base_classifier.preprocess:
        # using base_model (has no automatic pre_pocessing in __call__)
        preprocessing = dict(mean=base_classifier.mean, std=base_classifier.std, axis=-3)
    else:
        preprocessing = None

    fb_base_model = fb.PyTorchModel(base_model, bounds=(0, 1), device=device,
                                    preprocessing=preprocessing)

    # HSJA is a SOTA decision based attack
    attack = fb.attacks.HopSkipJumpAttack(steps=4,  # TODO use 32 or 64
                                          initial_gradient_eval_steps=64,
                                          max_gradient_eval_steps=1024,  # TODO 4096 should be fine
                                          gamma=1e4)  # defines threshold for binary search

    base_success_rate = torch.empty((len(bounds_l2), 0), device=device)
    def_success_rate = torch.empty((len(bounds_l2), 0), device=device)

    for b_idx, (x1, y1, x2, y2) in enumerate(tqdm(dataloader)):

        # if b_idx == 2:  # TODO REMOVE
        #     break

        x1 = torch.clip(x1.to(device), 0.0, 1.0)
        y1 = y1.to(device)
        x2 = torch.clip(x2.to(device), 0.0, 1.0)
        y2 = y2.to(device)

        # validate that base model predicts correct classes (remove wrong ones)
        preds_1 = base_classifier(x1).argmax(dim=1)
        preds_2 = base_classifier(x2).argmax(dim=1)
        passed = torch.logical_and(torch.eq(preds_1, y1), torch.eq(preds_2, y2))
        x1, x2, y1, y2 = x1[passed], x2[passed], y1[passed], y2[passed]

        # remove images where starting adv is not adversarial for def model
        flag = defense_model.preprocess
        defense_model.preprocess = True
        preds_2 = defense_model(x2).argmax(dim=1)
        passed = torch.eq(preds_2, y2)
        x1, x2, y1, y2 = x1[passed], x2[passed], y1[passed], y2[passed]
        defense_model.preprocess = flag

        # HSJA with increasing L2 bounds.
        # adv = list of len(bounds_l2) of tensors B C H W.
        # success = boolean tensor of shape (N_bounds, B)
        _, adv_base, suc_base = attack(fb_base_model, x1, y1, starting_points=x2, epsilons=bounds_l2)
        _, adv, suc = attack(fb_defense_model, x1, y1, starting_points=x2, epsilons=bounds_l2)

        # cat results
        base_success_rate, _ = pack([base_success_rate, suc_base], 'n *')
        def_success_rate, _ = pack([def_success_rate, suc_def], 'n *')

        # # save visual examples of applied perturbation.
        # if b_idx % 20 == 0:
        #
        #     # estimate of the cleaned image (estimate because the process is random)
        #     cleaned_imgs = [defense_model(found_adv, preds_only=False)[-1] for found_adv in adv]
        #
        #     for i, b in enumerate(bounds_l2):
        #
        #         max_b = 8
        #         img_i = images.clone()[:max_b]
        #         adv_i = adv[i][:max_b]
        #         def_i = cleaned_imgs[i][:max_b]
        #         max_b = adv_i.shape[0]
        #
        #         examples = torch.cat([img_i, adv_i, def_i], dim=0)
        #         display = make_grid(examples, nrow=max_b).permute(1, 2, 0).cpu()
        #         plt.imshow(display)
        #         plt.axis(False)
        #         plt.title(f'originals, adversarial and cleaned images at L2={b:.2f}')
        #         plt.savefig(f'{args.plots_folder}batch_{b_idx}_bound={b:.2f}.png')
        #         plt.close()

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

