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

    # load pre-trained models
    if args.classifier_type == 'resnet-32':
        base_classifier = Cifar10ResnetModel(args.classifier_path, device)
        defense_model = Cifar10NVAEDefenseModel(base_classifier, args.autoencoder_path, device, args.resample_from)
        args.image_size = 32
        bounds_l2 = (0.1, 0.2, 0.5, 1.0, 2.0)
    elif args.classifier_type == 'vgg-16':
        base_classifier = Cifar10VGGModel(args.classifier_path, device)
        defense_model = Cifar10NVAEDefenseModel(base_classifier, args.autoencoder_path, device, args.resample_from)
        args.image_size = 32
        bounds_l2 = (0.1, 0.2, 0.5, 1.0, 2.0)  # TODO check
    elif args.classifier_type == 'resnet-50':
        base_classifier = CelebAResnetModel(args.classifier_path, device)
        defense_model = CelebAStyleGanDefenseModel(base_classifier, args.autoencoder_path, device, args.resample_from)
        args.image_size = 256
        bounds_l2 = (0.25, 0.5, 1.0, 2.0)
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
    fb_defense_model = fb.PyTorchModel(defense_model, bounds=(0, 1), device=device,
                                       preprocessing=preprocessing)
    fb_defense_model = fb.models.ExpectationOverTransformationWrapper(fb_defense_model, n_steps=8)  # TODO check

    if base_classifier.preprocess:
        # using base_model (has no automatic pre_pocessing in __call__)
        preprocessing = dict(mean=base_classifier.mean, std=base_classifier.std, axis=-3)
    else:
        preprocessing = None

    fb_base_model = fb.PyTorchModel(base_model, bounds=(0, 1), device=device,
                                    preprocessing=preprocessing)

    # select attack
    if args.attack == 'C&W':
        # attack = fb.attacks.L2CarliniWagnerAttack(binary_search_steps=9, steps=10000)
        attack = fb.attacks.L2CarliniWagnerAttack(binary_search_steps=2, steps=2048)
    else:
        # DeepFool with base params
        attack = fb.attacks.L2DeepFoolAttack(steps=50,  candidates=10, overshoot=0.02)

    base_success_rate = torch.empty((len(bounds_l2), 0), device=device)
    def_success_rate = torch.empty((len(bounds_l2), 0), device=device)

    for b_idx, (x1, y1, _, _) in enumerate(tqdm(dataloader)):

        # if b_idx == 4:  # TODO REMOVE
        #     break

        # IF C&W, do only 1/10 of steps
        if args.attack == 'C&W' and b_idx % 10 != 0:
            continue

        with torch.no_grad():

            x1 = torch.clip(x1.to(device), 0.0, 1.0)
            y1 = y1.to(device)

            # validate that base model predicts correct classes (remove wrong ones)
            preds_1 = base_classifier(x1).argmax(dim=1)
            passed = torch.eq(preds_1, y1)
            x1, y1 = x1[passed], y1[passed]

            if passed.view(-1).sum().item() == 0:
                continue

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
                cleaned_imgs = [defense_model(found_adv, preds_only=False)[-1] for found_adv in adv_def]

                for i, b in enumerate(bounds_l2):

                    max_b = 8
                    img_i = x1.clone()[:max_b]
                    adv_i = adv_def[i][:max_b]
                    def_i = cleaned_imgs[i][:max_b]
                    max_b = adv_i.shape[0]

                    examples = torch.cat([img_i, adv_i, def_i], dim=0)
                    display = make_grid(examples, nrow=max_b).permute(1, 2, 0).cpu()
                    plt.imshow(display)
                    plt.axis(False)
                    plt.title(f'originals, adversarial and cleaned images at L2={b:.2f}')
                    plt.savefig(f'{args.plots_folder}/bound={b:.2f}_batch_{b_idx}.png')
                    plt.close()

    # compute and save final results
    res_dict = {}

    for i, b in enumerate(bounds_l2):
        # print(f'brute force l2 bound={b:.2f} success base {base_success_rate[i].mean().item()}')
        # print(f'brute force l2 bound={b:.2f} success defense {def_success_rate[i].mean().item()}')

        res_dict[f'l2 bound={b:.2f} success on base'] = base_success_rate[i].mean().item()
        res_dict[f'l2 bound={b:.2f} success on defense'] = def_success_rate[i].mean().item()

    with open(f'{args.results_folder}results_{args.attack}.json', 'w') as f:
         json.dump(res_dict, f)


if __name__ == '__main__':

    arguments = parse_args()
    main(arguments)

