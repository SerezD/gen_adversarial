import argparse
import json

import foolbox as fb
import torch.multiprocessing as mp
import os
import torch
from torch import nn
from torch._C._distributed_c10d import ReduceOp
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Normalize
from tqdm import tqdm

from data.datasets import ImageLabelDataset
from src.defenses.common_utils import load_ResNet_CelebA, load_hub_CNN
from src.experiments.competitors.nd_vae.src.models.NVAE import Defence_NVAE
from src.experiments.competitors.nd_vae.src.models.NVAE_utils import DiscMixLogistic


class DefenseModel(nn.Module):

    def __init__(self, base_classifier, purifier, classifier_preprocessing = None):
        super().__init__()

        self.base_classifier = base_classifier
        self.purifier = purifier
        self.classifier_preprocessing = classifier_preprocessing

    def forward(self, x):

        # add gaussian noise with std = 0.1
        x = x + torch.randn_like(x) * 0.1
        x = torch.clamp(x, 0, 1)

        logits = self.purifier(x)[0]
        x_cln = DiscMixLogistic(logits).mean()

        if self.classifier_preprocessing is not None:
            x_cln = self.classifier_preprocessing(x_cln)

        preds = self.base_classifier(x_cln)
        return preds


def ddp_setup(rank: int, world_size: int):

   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"

   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size)


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


def main(rank, world_size, args):

    ddp_setup(rank, world_size)

    device = f'cuda:{rank}'

    # load pre-trained models
    if args.classifier_type == 'resnet-32':
        base_classifier = load_hub_CNN(args.classifier_path, args.classifier_type, device)
        nd_vae = None  # TODO
        classifier_preprocessing = None
        args.image_size = 32
        bounds_l2 = (0.5, 1.0, 2.0, 4.0)
    elif args.classifier_type == 'vgg-16':
        base_classifier = load_hub_CNN(args.classifier_path, args.classifier_type, device)
        nd_vae = None  # TODO
        classifier_preprocessing = None
        args.image_size = 32
        bounds_l2 = (0.5, 1.0, 2.0, 4.0)
    elif args.classifier_type == 'resnet-50':
        base_classifier = load_ResNet_CelebA(args.classifier_path, device)

        args.image_size = 256
        bounds_l2 = (0.5, 1.0, 2.0, 4.0)  # TODO CHECK BOUNDS
        classifier_preprocessing = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        NVAE_PARAMS = {
            'x_channels': 3,
            'pre_proc_groups': 2,
            'encoding_channels': 16,
            'scales': 2,
            'groups': 4,
            'cells': 2
        }

        nd_vae = Defence_NVAE(NVAE_PARAMS['x_channels'],
                         NVAE_PARAMS['encoding_channels'],
                         NVAE_PARAMS['pre_proc_groups'],
                         NVAE_PARAMS['scales'],
                         NVAE_PARAMS['groups'],
                         NVAE_PARAMS['cells'],
                         args.image_size)

        state_dict = torch.load(args.autoencoder_path)
        nd_vae.load_state_dict(state_dict)
        nd_vae.to(device).eval()

    else:
        raise ValueError(f'Unknown classifier type: {args.classifier_type}')

    defense_model = DefenseModel(base_classifier, nd_vae, classifier_preprocessing).eval()

    # wrap defense with EOT to maintain deterministic predictions
    # this is a simple counter_attack, to avoid gradient masking
    fb_defense_model = fb.PyTorchModel(defense_model, bounds=(0, 1), device=device)
    fb_defense_model = fb.models.ExpectationOverTransformationWrapper(fb_defense_model, n_steps=8)

    # dataloader
    dataset = ImageLabelDataset(folder=args.images_path, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=DistributedSampler(dataset))
    dataloader.sampler.set_epoch(0)

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

    success_rates = torch.empty((3, len(bounds_l2), 0), device=device)

    for idx, (images, labels) in enumerate(tqdm(dataloader)):

        # if idx > 14:
        #     break

        if idx % n_steps != 0:
            continue

        images = torch.clip(images.to(device), 0.0, 1.0)
        labels = labels.to(device)

        # uncomment to visualize some examples
        # from torchvision.utils import make_grid
        # from matplotlib import pyplot as plt
        # images += torch.randn_like(images) * 0.1
        # logits = nd_vae(images)[0]
        # x_cln = DiscMixLogistic(logits).mean()
        # display = make_grid(torch.cat((images, x_cln), dim=0), nrow=args.batch_size).permute(1, 2, 0).cpu().numpy()
        # plt.imshow(display)
        # plt.show()
        # continue

        # CLEAN ACCURACY
        with torch.no_grad():
            # Note: success rate is true if prediction is NOT EQUAL to label!
            success_0 = torch.ne(fb_defense_model(images).argmax(dim=1), labels)
            success_0 = success_0.unsqueeze(0).repeat(len(bounds_l2), 1)

        # DEEP FOOL
        success_1 = attacks[0](fb_defense_model, images, labels, epsilons=bounds_l2)[2]

        # C&W
        success_2 = attacks[1](fb_defense_model, images, labels, epsilons=bounds_l2)[2]

        success_rates = torch.cat((success_rates,
                                   torch.stack((success_0, success_1, success_2), dim=0)), dim=2)

    # Merge results of different ranks
    success_rates = success_rates.mean(dim=-1)  # n_attacks, n_bounds
    torch.distributed.all_reduce(success_rates, ReduceOp.SUM)
    success_rates = success_rates / world_size

    if rank == 0:
        res_dict = {}
        for a_i, attack in enumerate(['Clean', 'DeepFool', 'C&W']):
            for b_i, bound in enumerate(bounds_l2):
                res_dict[f'{attack}_{bound:.2f}'] = success_rates[a_i][b_i].item()

        with open(f'{args.results_folder}results.json', 'w') as f:
            json.dump(res_dict, f)

    destroy_process_group()


if __name__ == '__main__':

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, parse_args()), nprocs=world_size)