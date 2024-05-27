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
from src.experiments.competitors.a_vae.model import StyledGenerator


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


def ddp_setup(rank: int, world_size: int):

   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"

   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main(rank, world_size, args):

    ddp_setup(rank, world_size)

    device = f'cuda:{rank}'

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
    dataset = ImageLabelDataset(folder=args.images_path, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=DistributedSampler(dataset))
    dataloader.sampler.set_epoch(0)

    # attacks
    n_steps = 15  # reduce number of tests
    attacks = [
        fb.attacks.L2DeepFoolAttack(steps=50,  candidates=10, overshoot=0.02),
        # TODO TEST
        fb.attacks.L2CarliniWagnerAttack(binary_search_steps=2,
                                         steps=2048,
                                         confidence=2e-2,
                                         initial_const=16.0)
    ]

    success_rates = torch.empty((3, len(bounds_l2), 0), device=device)

    for idx, (images, labels) in enumerate(tqdm(dataloader)):

        # TODO
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
