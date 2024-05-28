import argparse
import json

import foolbox as fb
import torch.multiprocessing as mp
import os
import torch
from matplotlib import pyplot as plt
from torch._C._distributed_c10d import ReduceOp
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid

from tqdm import tqdm

from data.datasets import ImageLabelDataset
from src.defenses.models import CelebAResnetModel, CelebAStyleGanDefenseModel


def parse_args():

    parser = argparse.ArgumentParser('Test the HL purification defense model')

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

    parser.add_argument('--interpolation_alphas', type=float, required=True, nargs='+',
                        help='For each hierarchy, degree of interpolation between reconstruction code '
                             '(alpha=0) and new sampled code (alpha=1).')

    parser.add_argument('--initial_noise_eps', type=float, default=0.,
                        help='l2 bound for perturbation of initial images (before purification)')

    parser.add_argument('--gaussian_blur_input', action='store_true',
                        help='l2 bound for perturbation of initial images (before purification)')

    parser.add_argument('--results_folder', type=str, required=True,
                        help='folder to save .pickle file with results')

    args = parser.parse_args()

    # create folder
    args.results_folder = (f'{args.results_folder}/'
                           f'{args.autoencoder_name}_noise:{args.initial_noise_eps}_blur:{args.gaussian_blur_input}_'
                           f'{args.classifier_type}/')

    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    args.plots_folder = f'{args.results_folder}/plots/'
    if not os.path.exists(args.plots_folder):
        os.makedirs(args.plots_folder)

    return args


def ddp_setup(rank: int, world_size: int):

   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"

   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main(rank, world_size, args):

    ddp_setup(rank, world_size)

    device = f'cuda:{rank}'

    if args.classifier_type == 'resnet-50':
        bounds_l2 = (0.5, 1.0, 2.0, 4.0)
        args.image_size = 256
        base_classifier = CelebAResnetModel(args.classifier_path, device)
        defense_model = CelebAStyleGanDefenseModel(base_classifier, args.autoencoder_path,
                                                   args.interpolation_alphas, args.initial_noise_eps,
                                                   args.gaussian_blur_input, device)
    else:
        raise ValueError(f'Unknown classifier type: {args.classifier_type}')

    # dataloader
    dataset = ImageLabelDataset(folder=args.images_path, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=DistributedSampler(dataset))
    dataloader.sampler.set_epoch(0)

    # base classifier to measure undefended performances
    fb_base_model = fb.PyTorchModel(base_classifier.eval(), bounds=(0, 1), device=device)

    # wrap defense with EOT to maintain deterministic predictions
    # this is a simple counter_attack, to avoid gradient masking
    # TODO ensure to maintain the same n_steps in all cases
    fb_defense_model = fb.PyTorchModel(defense_model.eval(), bounds=(0, 1), device=device)
    fb_defense_model = fb.models.ExpectationOverTransformationWrapper(fb_defense_model, n_steps=8)

    # attacks
    n_steps = 15  # reduce number of tests
    attacks = [
        fb.attacks.L2DeepFoolAttack(steps=50, candidates=10, overshoot=0.02),
        # TODO TEST
        fb.attacks.L2CarliniWagnerAttack(binary_search_steps=2,
                                         steps=2048,
                                         confidence=2e-2,
                                         initial_const=16.0)
    ]

    success_rates_base = torch.empty((3, len(bounds_l2), 0), device=device)
    success_rates_defense = torch.empty((3, len(bounds_l2), 0), device=device)

    for idx, (images, labels) in enumerate(tqdm(dataloader)):

        # TODO
        # if idx > 14:
        #     break

        if idx % n_steps != 0:
            continue

        images = torch.clip(images.to(device), 0.0, 1.0)
        labels = labels.to(device)

        # classifier only
        with torch.no_grad():
            success_0 = torch.ne(fb_base_model(images).argmax(dim=1), labels)
            success_0 = success_0.unsqueeze(0).repeat(len(bounds_l2), 1)

        # DEEP FOOL and C&W
        _, _, success_1 = attacks[0](fb_base_model, images, labels, epsilons=bounds_l2)
        _, _, success_2 = attacks[1](fb_base_model, images, labels, epsilons=bounds_l2)

        # cat results
        success_rates_base = torch.cat((success_rates_base,
                                   torch.stack((success_0, success_1, success_2), dim=0)), dim=2)

        # Defense
        with torch.no_grad():
            success_0 = torch.ne(fb_defense_model(images).argmax(dim=1), labels)
            success_0 = success_0.unsqueeze(0).repeat(len(bounds_l2), 1)

        # DEEP FOOL and C&W
        _, adv_df, success_1 = attacks[0](fb_defense_model, images, labels, epsilons=bounds_l2)
        _, adv_cw, success_2 = attacks[1](fb_defense_model, images, labels, epsilons=bounds_l2)

        # cat results
        success_rates_defense = torch.cat((success_rates_defense,
                                        torch.stack((success_0, success_1, success_2), dim=0)), dim=2)

        # save visual examples of applied perturbation.
        if rank == 0:

            with torch.no_grad():

                for (adv, name) in zip([adv_df, adv_cw], ['deep_fool', 'C&W']):

                    # estimate of the cleaned image (estimate because the process is random)
                    cleaned_imgs = [defense_model(found_adv, preds_only=False)[-1] for found_adv in adv]

                    for i, b in enumerate(bounds_l2):
                        max_b = 8
                        img_i = images.clone()[:max_b].clamp(0., 1.)
                        adv_i = adv[i][:max_b].clamp(0., 1.)
                        def_i = cleaned_imgs[i][:max_b].clamp(0., 1.)
                        max_b = adv_i.shape[0]

                        examples = torch.cat([img_i, adv_i, def_i], dim=0)
                        display = make_grid(examples, nrow=max_b).permute(1, 2, 0).cpu()
                        plt.imshow(display)
                        plt.axis(False)
                        plt.title(f'originals, adversarial and cleaned images at L2={b:.2f}')
                        plt.savefig(f'{args.plots_folder}/{name}_bound={b:.2f}_batch={idx}.png')
                        plt.close()

    #  Merge results of different ranks
    success_rates_base = success_rates_base.mean(dim=-1)  # n_attacks, n_bounds
    torch.distributed.all_reduce(success_rates_base, ReduceOp.SUM)
    success_rates_base = success_rates_base / world_size

    success_rates_defense = success_rates_defense.mean(dim=-1)  # n_attacks, n_bounds
    torch.distributed.all_reduce(success_rates_defense, ReduceOp.SUM)
    success_rates_defense = success_rates_defense / world_size

    if rank == 0:

        # BASE
        res_dict = {}
        for a_i, attack in enumerate(['Clean', 'DeepFool', 'C&W']):
            for b_i, bound in enumerate(bounds_l2):
                res_dict[f'{attack}_{bound:.2f}'] = success_rates_base[a_i][b_i].item()

        with open(f'{args.results_folder}base_results.json', 'w') as f:
            json.dump(res_dict, f)

        # DEFENSE
        res_dict = {}
        for a_i, attack in enumerate(['Clean', 'DeepFool', 'C&W']):
            for b_i, bound in enumerate(bounds_l2):
                res_dict[f'{attack}_{bound:.2f}'] = success_rates_defense[a_i][b_i].item()

        with open(f'{args.results_folder}def_results.json', 'w') as f:
            json.dump(res_dict, f)

    destroy_process_group()


if __name__ == '__main__':

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, parse_args()), nprocs=world_size)
