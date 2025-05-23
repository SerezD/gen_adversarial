import argparse
import json
import numpy as np
import os
import random
import torch

from datetime import timedelta
from matplotlib import pyplot as plt
from torch import multiprocessing as mp
from torch.cuda import synchronize
from torch.distributed import init_process_group, destroy_process_group, barrier
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid
from tqdm import tqdm

from data.datasets import ImageLabelDataset
from src.experiments.load_defense import load


def pad_image(image: torch.Tensor, color: bool, padding_size: int = 2) -> torch.Tensor:
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


def parse_args() -> args.Namespace:

    parser = argparse.ArgumentParser('Common Pipeline to test a given defense mechanism.')

    parser.add_argument('--images_path', type=str, required=True,
                        help='All images in this folder will be attacked')

    parser.add_argument('--eot_steps', type=int, default=32)

    parser.add_argument('--defense_type', type=str, choices=['base', 'A-VAE', 'ND-VAE', 'trades',
                                                             'ours', 'ablation'])

    parser.add_argument('--experiment', type=str, choices=['gender', 'ids', 'cars'])

    parser.add_argument('--config', type=str, required=True,
                        help='Path to .yaml configuration file with the Defense Model information.')

    parser.add_argument('--attack', type=str, choices=['deepfool', 'c&w', 'autoattack'], default=None,
                        help='If passed, try a specific attack only. Otherwise, try all.')

    args = parser.parse_args()

    # create results folder
    args.results_folder = f'./results/{args.config.split("/")[-1][:-5]}/'

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


def main(rank: int, world_size: int, args: args.Namespace):

    ddp_setup(rank, world_size)

    args.device = f'cuda:{rank}'

    args, defense_model = load(args)

    # dataloader
    dataset = ImageLabelDataset(folder=args.images_path, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=DistributedSampler(dataset, shuffle=False))

    clean_accuracy = torch.empty((0, 1), device=args.device)
    df_distortions = torch.empty((0, 1), device=args.device)
    cw_distortions = torch.empty((0, 1), device=args.device)
    aa_distortions = torch.empty((0, 1), device=args.device)

    for idx, (images, labels) in enumerate(tqdm(dataloader)):

        # Ensure all processes have finished previous computations before to continue
        synchronize(rank)
        barrier()

        images = torch.clamp(images.to(args.device), 0.0, 1.0)
        labels = labels.to(args.device)

        # test "no attack"
        with torch.no_grad():
            preds = defense_model(images)
            clean_accuracy = torch.cat((clean_accuracy, torch.eq(preds.argmax(dim=1), labels).unsqueeze(0)), 0)

        # DEEP FOOL
        if args.attack is None or args.attack == 'deepfool':
            succeeded_df, minimum_bound_df, adversarial_df = args.attacks['deepfool'](images, labels, defense_model)

            if succeeded_df:
                df_distortions = torch.cat((df_distortions,
                                            torch.tensor([[minimum_bound_df]], device=args.device)))
            else:
                df_distortions = torch.cat((df_distortions,
                                            torch.tensor([[100.]], device=args.device)))

        else:
            if rank == 0 and idx == 0:
                print('Skipping DeepFool')

            df_distortions = torch.cat((df_distortions, torch.tensor([[0.]], device=args.device)))
            succeeded_df = False
            minimum_bound_df = 0.
            adversarial_df = images.detach().clone()

        # C&W
        if args.attack is None or args.attack == 'c&w':

            succeeded_cw, minimum_bound_cw, adversarial_cw = args.attacks['c&w'](images, labels, defense_model)

            if succeeded_cw:
                cw_distortions = torch.cat((cw_distortions,
                                            torch.tensor([[minimum_bound_cw]], device=args.device)))
            else:
                cw_distortions = torch.cat((cw_distortions,
                                            torch.tensor([[100.]], device=args.device)))

        else:

            if rank == 0 and idx == 0:
                print('Skipping C&W')

            cw_distortions = torch.cat((cw_distortions, torch.tensor([[0.]], device=args.device)))
            succeeded_cw = False
            minimum_bound_cw = 0.0
            adversarial_cw = images.detach().clone()

        # C&W
        if args.attack is None or args.attack == 'autoattack':

            succeeded_aa, minimum_bound_aa, adversarial_aa = args.attacks['autoattack'](images, labels, defense_model)

            if succeeded_aa:
                aa_distortions = torch.cat((aa_distortions,
                                            torch.tensor([[minimum_bound_aa]], device=args.device)))
            else:
                aa_distortions = torch.cat((aa_distortions,
                                            torch.tensor([[100.]], device=args.device)))

        else:

            if rank == 0 and idx == 0:
                print('Skipping AutoAttack')

            aa_distortions = torch.cat((aa_distortions, torch.tensor([[0.]], device=args.device)))
            succeeded_aa = False
            minimum_bound_aa = 0.0
            adversarial_aa = images.detach().clone()

        # save visual examples of applied perturbation.
        if rank == 0 and idx % 5 == 0:

            with torch.no_grad():

                # add padding to base image
                pad_size = int(np.log2(args.image_size))
                images = pad_image(images, False, pad_size)

                for (success, bound, adv_example, name) in zip([succeeded_df, succeeded_cw, succeeded_aa],
                                                               [minimum_bound_df, minimum_bound_cw, minimum_bound_aa],
                                                               [adversarial_df, adversarial_cw, adversarial_aa],
                                                               ['deep_fool', 'C&W', 'AutoAttack']):

                    # skip attack if not computed.
                    if args.attack is not None:
                        if name == 'deep_fool' and args.attack != 'deepfool':
                            continue
                        elif name == 'C&W' and args.attack != 'c&w':
                            continue
                        elif name == 'AutoAttack' and args.attack != 'autoattack':
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
    gathered_clean_accuracy = [torch.zeros_like(clean_accuracy) for _ in range(world_size)]
    gathered_df_distortions = [torch.zeros_like(df_distortions) for _ in range(world_size)]
    gathered_cw_distortions = [torch.zeros_like(cw_distortions) for _ in range(world_size)]
    gathered_aa_distortions = [torch.zeros_like(aa_distortions) for _ in range(world_size)]

    torch.distributed.all_gather(gathered_clean_accuracy, clean_accuracy)
    torch.distributed.all_gather(gathered_df_distortions, df_distortions)
    torch.distributed.all_gather(gathered_cw_distortions, cw_distortions)
    torch.distributed.all_gather(gathered_aa_distortions, aa_distortions)

    clean_accuracy = torch.cat(gathered_clean_accuracy, dim=0).squeeze(1).mean(dim=0).item()
    df_distortions = torch.cat(gathered_df_distortions, dim=0).squeeze(1).cpu().tolist()
    cw_distortions = torch.cat(gathered_cw_distortions, dim=0).squeeze(1).cpu().tolist()
    aa_distortions = torch.cat(gathered_aa_distortions, dim=0).squeeze(1).cpu().tolist()

    if rank == 0:

        json_dest_file = f'{args.results_folder}results.json'

        # load previous result, if exists
        if os.path.exists(json_dest_file):
            with open(json_dest_file, 'r') as file:
                res_dict = json.load(file)
        else:
            res_dict = {}

        # update, write on dict
        for a_i, attack in enumerate(['Clean', 'DeepFool', 'C&W', 'AutoAttack']):

            # SKIP If not computed
            if args.attack is not None:
                if attack == 'DeepFool' and args.attack != 'deepfool':
                    continue
                elif attack == 'C&W' and args.attack != 'c&w':
                    continue
                elif attack == 'AutoAttack' and args.attack != 'autoattack':
                    continue

            if attack == 'Clean':
                value = clean_accuracy
            elif attack == 'DeepFool':
                value = df_distortions
            elif attack == 'C&W':
                value = cw_distortions
            else:
                value = aa_distortions

            res_dict[f'{attack}'] = value

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
