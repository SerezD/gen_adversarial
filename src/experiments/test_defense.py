import argparse
import json
import random
import numpy as np
import torch.multiprocessing as mp
import os
from datetime import timedelta
import torch
from matplotlib import pyplot as plt
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group, barrier
from torch.cuda import synchronize
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid

from tqdm import tqdm

from data.datasets import ImageLabelDataset
from src.experiments.load_defense import load


def pad_image(image, color, padding_size: int = 2):
    """
    pad image (1, 3, h, w) on all sides
    with color red (if color is true) or green
    """

    color = (1., 0., 0.) if color else (0., 1., 0.)

    image = pad(image, (padding_size, padding_size, padding_size, padding_size))

    # Now manually color
    image[:, 0, :padding_size, :] = color[0]  # Top padding
    image[:, 1, :padding_size, :] = color[1]
    image[:, 2, :padding_size, :] = color[2]

    image[:, 0, -padding_size:, :] = color[0]  # Bottom padding
    image[:, 1, -padding_size:, :] = color[1]
    image[:, 2, -padding_size:, :] = color[2]

    image[:, 0, :, :padding_size] = color[0]  # Left padding
    image[:, 1, :, :padding_size] = color[1]
    image[:, 2, :, :padding_size] = color[2]

    image[:, 0, :, -padding_size:] = color[0]  # Right padding
    image[:, 1, :, -padding_size:] = color[1]
    image[:, 2, :, -padding_size:] = color[2]

    return image


def parse_args():

    parser = argparse.ArgumentParser('Common Pipeline to test a given defense mechanism.')

    parser.add_argument('--images_path', type=str, required=True,
                        help='All images in this folder will be attacked')

    parser.add_argument('--eot_steps', type=int, default=32)

    parser.add_argument('--defense_type', type=str, choices=['base', 'A-VAE', 'ND-VAE', 'trades',
                                                             'ours', 'ablation'])

    parser.add_argument('--experiment', type=str, choices=['gender', 'ids', 'cars'])

    parser.add_argument('--config', type=str, required=True,
                        help='Path to .yaml configuration file with the Defense Model information.')

    parser.add_argument('--attack', type=str, choices=['deepfool', 'c&w'], default=None,
                        help='If passed, try a specific attack only. Otherwise, try all.')

    args = parser.parse_args()

    # create results folder
    args.results_folder = f'./results/{args.config.split('/')[-1][:-5]}/'

    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    args.plots_folder = f'{args.results_folder}/plots/'
    if not os.path.exists(args.plots_folder):
        os.makedirs(args.plots_folder)

    return args


def ddp_setup(rank: int, world_size: int):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # ensure reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(hours=12))


def main(rank, world_size, args):

    ddp_setup(rank, world_size)

    args.device = f'cuda:{rank}'

    args, defense_model = load(args)

    # dataloader
    dataset = ImageLabelDataset(folder=args.images_path, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=DistributedSampler(dataset, shuffle=False))

    success_rates = torch.empty((3, len(args.bounds_l2), 0), device=args.device)

    for idx, (images, labels) in enumerate(tqdm(dataloader)):

        # Ensure all processes have finished previous computations before to continue
        synchronize(rank)
        barrier()

        images = torch.clamp(images.to(args.device), 0.0, 1.0)
        labels = labels.to(args.device)

        # test "no attack"
        with torch.no_grad():
            preds = defense_model(images)
            success_0 = torch.ne(preds.argmax(dim=1), labels)
            success_0 = success_0.unsqueeze(0).repeat(len(args.bounds_l2), 1)  # just copy on every bound

        # DEEP FOOL
        if args.attack is None or args.attack == 'deepfool':
            succeeded_df, minimum_bound_df, adversarial_df = args.attacks['deepfool'](images, labels, defense_model)
            success_1 = torch.zeros_like(success_0, dtype=torch.bool)
            for i, b in enumerate(args.bounds_l2):
                if succeeded_df and minimum_bound_df <= b:
                    success_1[i] = True
        else:
            if rank == 0 and idx == 0:
                print('Skipping DeepFool')
            success_1 = torch.zeros_like(success_0, dtype=torch.bool)
            succeeded_df = False
            minimum_bound_df = 0.
            adversarial_df = images.detach().clone()

        # C&W
        if args.attack is None or args.attack == 'c&w':

            succeeded_cw, minimum_bound_cw, adversarial_cw = args.attacks['cw'](images, labels, defense_model)
            success_2 = torch.zeros_like(success_0, dtype=torch.bool)
            for i, b in enumerate(args.bounds_l2):
                if succeeded_cw and minimum_bound_cw <= b:
                    success_2[i] = True
        else:

            if rank == 0 and idx == 0:
                print('Skipping C&W')

            success_2 = torch.zeros_like(success_0, dtype=torch.bool)
            succeeded_cw = False
            minimum_bound_cw = 0.0
            adversarial_cw = images.detach().clone()

        # cat results
        success_rates = torch.cat((success_rates,
                                   torch.stack((success_0, success_1, success_2), dim=0)), dim=2)

        # save visual examples of applied perturbation.
        if rank == 0 and idx % 5 == 0:

            with torch.no_grad():

                # add padding to base image
                pad_size = int(np.log2(args.image_size))
                images = pad_image(images, False, pad_size)

                for (success, bound, adv_example, name) in zip([succeeded_df, succeeded_cw],
                                                               [minimum_bound_df, minimum_bound_cw],
                                                               [adversarial_df, adversarial_cw], ['deep_fool', 'C&W']):

                    # skip attack if not computed.
                    if args.attack is not None:
                        if name == 'deep_fool' and args.attack != 'deepfool':
                            continue
                        elif name == 'C&W' and args.attack != 'c&w':
                            continue

                    # estimate of the cleaned image (estimate because the process is random)
                    cleaned_sample = defense_model.get_purified(adv_example).clamp(0., 1.)

                    # add padding to other images
                    adv_example = pad_image(adv_example.clamp(0., 1.), True, pad_size)
                    cleaned_sample = pad_image(cleaned_sample, success, pad_size)

                    examples = torch.cat([images, adv_example, cleaned_sample], dim=0)
                    display = make_grid(examples, padding=0, nrow=3).permute(1, 2, 0).cpu()
                    plt.imshow(display)
                    plt.axis(False)
                    plt.title(f'originals, adversarial [L2={bound:.2f}] and cleaned images')
                    plt.savefig(f'{args.plots_folder}/{name}_example={idx}.png')
                    plt.close()

    #  Gather results of different ranks
    gathered_success_rates = [torch.zeros_like(success_rates) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_success_rates, success_rates)
    gathered_success_rates = torch.cat(gathered_success_rates, dim=1)
    success_rates = gathered_success_rates.mean(dim=-1)

    if rank == 0:

        json_dest_file = f'{args.results_folder}results.json'

        # load previous result, if exists
        if os.path.exists(json_dest_file):
            with open(json_dest_file, 'r') as file:
                res_dict = json.load(file)
        else:
            res_dict = {}

        # update, write on dict
        for a_i, attack in enumerate(['Clean', 'DeepFool', 'C&W']):

            if args.attack is not None:
                if attack == 'DeepFool' and args.attack != 'deepfool':
                    continue
                elif attack == 'C&W' and args.attack != 'c&w':
                    continue

            for b_i, bound in enumerate(args.bounds_l2):
                if attack == 'Clean':
                    res_dict[f'{attack}'] = success_rates[a_i][b_i].item()
                    break
                res_dict[f'{attack}_{bound}'] = success_rates[a_i][b_i].item()

        # write or overwrite/update
        with open(json_dest_file, 'w') as f:
            json.dump(res_dict, f)

    destroy_process_group()


if __name__ == '__main__':
    """
    Common Pipeline to test a given defense mechanism.
    """

    w_size = torch.cuda.device_count()
    mp.spawn(main, args=(w_size, parse_args()), nprocs=w_size)
