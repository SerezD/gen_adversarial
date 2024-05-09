import argparse
import json
import os
import torch
from einops import pack
import foolbox as fb
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from data.datasets import ImageLabelDataset
from src.defenses.models import Cifar10ResnetModel, CelebAResnetModel, Cifar10VGGModel, Cifar10NVAEDefenseModel, \
    CelebAStyleGanDefenseModel


def parse_args():

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

    parser.add_argument('--attack', type=str, choices=['DeepFool', 'C&W'])

    args = parser.parse_args()

    # create folder
    args.results_folder = f'{args.results_folder}/{args.autoencoder_name}_{args.classifier_type}/'
    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    args.plots_folder = f'{args.results_folder}/plots_{args.attack}/'
    if not os.path.exists(args.plots_folder):
        os.makedirs(args.plots_folder)

    return args


def main(args: argparse.Namespace):
    """

    """

    device = f'cuda:0'
    torch.cuda.set_device(0)

    # load pre-trained models
    if args.classifier_type == 'resnet-32':
        bounds_l2 = (0.5, 1.0, 2.0, 4.0)
        gaussian_eps = bounds_l2[-1]
        args.image_size = 32
        base_classifier = Cifar10ResnetModel(args.classifier_path, device)
        defense_model = Cifar10NVAEDefenseModel(base_classifier, args.autoencoder_path, device, args.resample_from,
                                                gaussian_eps=gaussian_eps)
    elif args.classifier_type == 'vgg-16':
        bounds_l2 = (0.5, 1.0, 2.0, 4.0)
        gaussian_eps = bounds_l2[-1]
        args.image_size = 32
        base_classifier = Cifar10VGGModel(args.classifier_path, device)
        defense_model = Cifar10NVAEDefenseModel(base_classifier, args.autoencoder_path, device, args.resample_from,
                                                gaussian_eps=gaussian_eps)
    elif args.classifier_type == 'resnet-50':
        bounds_l2 = (0.5, 1.0, 2.0, 4.0)
        gaussian_eps = bounds_l2[-1]
        args.image_size = 256
        base_classifier = CelebAResnetModel(args.classifier_path, device)
        defense_model = CelebAStyleGanDefenseModel(base_classifier, args.autoencoder_path, device, args.resample_from,
                                                   gaussian_eps=gaussian_eps)
    else:
        raise ValueError(f'Unknown classifier type: {args.classifier_type}')

    # dataloader
    dataloader = DataLoader(ImageLabelDataset(folder=args.images_path, image_size=args.image_size),
                            batch_size=args.batch_size, shuffle=False)

    # base classifier to measure undefended performances
    fb_base_model = fb.PyTorchModel(base_classifier.eval(), bounds=(0, 1), device=device)

    # wrap defense with EOT to maintain deterministic predictions
    # this is a simple counter_attack, to avoid gradient masking
    # TODO ensure to maintain the same n_steps in all cases
    fb_defense_model = fb.PyTorchModel(defense_model.eval(), bounds=(0, 1), device=device)
    fb_defense_model = fb.models.ExpectationOverTransformationWrapper(fb_defense_model, n_steps=8)

    # select attack
    if args.attack == 'C&W':
        # attack = fb.attacks.L2CarliniWagnerAttack(binary_search_steps=9, steps=10000)
        attack = fb.attacks.L2CarliniWagnerAttack(binary_search_steps=2, steps=2048)  # TODO CHECK
    else:
        # DeepFool with base params
        attack = fb.attacks.L2DeepFoolAttack(steps=50,  candidates=10, overshoot=0.02)

    base_success_rate = torch.empty((len(bounds_l2), 0), device=device)
    def_success_rate = torch.empty((len(bounds_l2), 0), device=device)

    n_steps = 15

    for b_idx, (x1, y1) in enumerate(tqdm(dataloader)):

        # if b_idx == 4:  # TODO REMOVE
        #     break

        if b_idx % n_steps != 0:
            continue

        x1 = torch.clip(x1.to(device), 0.0, 1.0)
        y1 = y1.to(device)

        # adv = list of len(bounds_l2) of tensors B C H W.
        # success = boolean tensor of shape (N_bounds, B)
        _, adv_base, suc_base = attack(fb_base_model, x1, y1, epsilons=bounds_l2)
        _, adv_def, suc_def = attack(fb_defense_model, x1, y1, epsilons=bounds_l2)

        # cat results
        base_success_rate, _ = pack([base_success_rate, suc_base], 'n *')
        def_success_rate, _ = pack([def_success_rate, suc_def], 'n *')

        # save visual examples of applied perturbation.
        if b_idx % 20 == 0:

            with torch.no_grad():

                # estimate of the cleaned image (estimate because the process is random)
                if defense_model.mean is not None:
                    defense_model.preprocess = True
                cleaned_imgs = [defense_model(found_adv, preds_only=False)[-1] for found_adv in adv_def]

                for i, b in enumerate(bounds_l2):

                    max_b = 8
                    img_i = x1.clone()[:max_b].clamp(0., 1.)
                    adv_i = adv_def[i][:max_b].clamp(0., 1.)
                    def_i = cleaned_imgs[i][:max_b].clamp(0., 1.)
                    max_b = adv_i.shape[0]

                    examples = torch.cat([img_i, adv_i, def_i], dim=0)
                    display = make_grid(examples, nrow=max_b).permute(1, 2, 0).cpu()
                    plt.imshow(display)
                    plt.axis(False)
                    plt.title(f'originals, adversarial and cleaned images at L2={b:.2f}')
                    plt.savefig(f'{args.plots_folder}/resample={args.resample_from}_bound={b:.2f}_batch={b_idx}.png')
                    plt.close()

    # compute and save final results
    res_dict = {}

    for i, b in enumerate(bounds_l2):

        res_dict[f'l2 bound={b:.2f} success on base'] = base_success_rate[i].mean().item()
        res_dict[f'l2 bound={b:.2f} success on defense'] = def_success_rate[i].mean().item()

    print('Opening file...')
    with open(f'{args.results_folder}results_{args.attack}_resample:{args.resample_from}.json', 'w') as f:
         json.dump(res_dict, f)


if __name__ == '__main__':

    arguments = parse_args()
    main(arguments)

