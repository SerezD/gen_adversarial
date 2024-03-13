import argparse
import os
import pickle

import foolbox as fb
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import ImageLabelDataset
from src.final_experiments.common_utils import load_ResNet_CelebA, load_hub_CNN


def parse_args():

    parser = argparse.ArgumentParser('Compute various attacks using FoolBox library')

    parser.add_argument('--images_path', type=str, required=True,
                        help='All images in this folder will be attacked')

    parser.add_argument('--batch_size', type=int, required=True)

    parser.add_argument('--attacked_net_path', type=str, required=True,
                        help='path to the pre-trained attacked CNNs')

    parser.add_argument('--attacked_net_type', type=str, choices=['resnet-32', 'vgg-16', 'resnet-50'],
                        help='name of CNN that has been evaluated')

    parser.add_argument('--results_folder', type=str, required=True,
                        help='folder to save .pickle file with results')

    args = parser.parse_args()

    # create folder
    args.results_folder = f'{args.results_folder}/attacks_to_{args.attacked_net_type}/'
    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    return args


def main(images_path: str, batch_size: int, attacked_net_path: str, attacked_net_type: str,
         out_folder: str, device: str = 'cuda:0'):

    # load correct cnn
    if attacked_net_type == 'resnet-50':
        base_net = load_ResNet_CelebA(attacked_net_path, device)
        preprocessing = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], axis=-3)
        image_size = 256

    else:
        base_net = load_hub_CNN(attacked_net_path, attacked_net_type, device)
        preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
        image_size = 32

    # bounds are before preprocessing!
    # preprocessing is normalization on the -3 axis (channel dims in pytorch images)
    attacked_model = fb.PyTorchModel(base_net, bounds=(0, 1), preprocessing=preprocessing, device=device)

    # dataloader
    dataloader = DataLoader(ImageLabelDataset(folder=images_path, image_size=image_size),
                            batch_size=batch_size, shuffle=False)

    # attacks and bounds
    attacks = [
        # L_2
        fb.attacks.L2DeepFoolAttack(),
        fb.attacks.L2BasicIterativeAttack(),
        fb.attacks.L2ProjectedGradientDescentAttack(),
        fb.attacks.L2FastGradientAttack(),
        # L_INF
        fb.attacks.LinfFastGradientAttack(),
        fb.attacks.LinfDeepFoolAttack(),
        fb.attacks.LinfBasicIterativeAttack(),
        fb.attacks.LinfProjectedGradientDescentAttack()
    ]

    attack_names = [
        # L_2
        'DeepFool_L2',
        'BIM_L2',
        'PGD_L2',
        'FGM_L2',
        # L_INF
        'FSGM_LInf',
        'DeepFool_LInf',
        'BIM_LInf',
        'PGD_LInf'
    ]
    bounds_l2 = (0.1, 0.3, 0.5, 0.7, 0.9)
    bounds_l_inf = (1/255, 2/255, 4/255, 8/255, 16/255)


    print('[INFO] Computing attacks on batches...')
    for (name, attack) in tqdm(zip(attack_names, attacks)):

        bounds = bounds_l2 if 'L2' in name else bounds_l_inf

        # create a unique dict to save results
        # formatted as:
        #        {bound: {'i': (np.array(3, H, W); float ) }; ...}
        #        where 'i' is the index sample in the original ordering; the array is the adversarial sample and float
        #        is 1 if attack is successful, 0 otherwise
        save_dict = {}

        for b in bounds:
            save_dict[f'{b:.3f}'] = {}

        # loop on all batches
        for batch_idx, (images, labels) in enumerate(dataloader):

            # if batch_idx == 2:
            #     break

            images, labels = torch.clip(images.to(device), 0., 1.), labels.to(device)
            _, adv, success = attack(attacked_model, images, labels, epsilons=bounds)

            for i, b in enumerate(bounds):

                adv_i, suc_i = adv[i], success[i]

                # save each adversarial and its success rate
                for img_idx, (a, s) in enumerate(zip(adv_i, suc_i)):

                    sample_n = f'{int(batch_idx * batch_size) + img_idx}'
                    save_dict[f'{b:.3f}'][sample_n] = (a.cpu().numpy(), 1. if s else 0.)

        # .pickle file for this attack
        file_name = f'{out_folder}{name}.pickle'
        with open(file_name, 'wb') as f:
            pickle.dump(save_dict, f)


if __name__ == '__main__':

    arguments = parse_args()
    main(
        images_path=arguments.images_path,
        batch_size=arguments.batch_size,
        attacked_net_path=arguments.attacked_net_path,
        attacked_net_type=arguments.attacked_net_type,
        out_folder=arguments.results_folder
    )
