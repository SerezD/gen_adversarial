import argparse
import os
import pickle

import foolbox as fb
import eagerpy as ep
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import ImageLabelDataset
from src.final_experiments.common_utils import load_ResNet_AFHQ_Wild, load_hub_CNN


def parse_args():

    parser = argparse.ArgumentParser('Compute various attacks using FoolBox library')

    parser.add_argument('--attacks_type', type=str,
                        choices=['whitebox-l2', 'whitebox-linf', 'blackbox'])

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


def main(attack_type: str, images_path: str, batch_size: int, attacked_net_path: str, attacked_net_type: str,
         out_folder: str, device: str = 'cuda:0'):

    # load correct cnn
    if attacked_net_type == 'resnet-50':
        attacked_model = load_ResNet_AFHQ_Wild(attacked_net_path, device)
        preprocessing = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], axis=-3)
        image_size = 256

    else:
        attacked_model = load_hub_CNN(attacked_net_path, attacked_net_type, device)
        preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
        image_size = 32

    attacked_model = fb.PyTorchModel(attacked_model, bounds=(0, 1), preprocessing=preprocessing, device=device)

    # dataloader
    dataloader = DataLoader(ImageLabelDataset(folder=images_path, image_size=image_size),
                            batch_size=batch_size, shuffle=False)

    # attacks and bounds
    if attack_type == 'whitebox-l2':
        attacks = [fb.attacks.L2DeepFoolAttack(),
                   fb.attacks.L2BasicIterativeAttack(), fb.attacks.L2ProjectedGradientDescentAttack(),
                   fb.attacks.L2FastGradientAttack()]
        attack_names = ['DeepFool_L2', 'BIM_L2', 'PGD_L2', 'FGM_L2']
        bounds = (0.1, 0.3, 0.5, 0.7, 0.9)

    elif attack_type == 'whitebox-linf':
        attacks = [fb.attacks.LinfFastGradientAttack(), fb.attacks.LinfDeepFoolAttack(),
                   fb.attacks.LinfBasicIterativeAttack(), fb.attacks.LinfProjectedGradientDescentAttack()]
        attack_names = ['FSGM_LInf', 'DeepFool_LInf', 'BIM_LInf', 'PGD_LInf']
        bounds = (1/255, 2/255, 4/255, 8/255, 16/255)
    else:
        # blackbox
        attacks = [fb.attacks.BoundaryAttack(), fb.attacks.HopSkipJumpAttack()]
        attack_names = ['Boundary_L2', 'HSJA_L2']
        bounds = (0.1, 0.3, 0.5, 0.7, 0.9)

    # create a unique dict to save results
    save_dict = {}
    for n in attack_names:
        for b in bounds:
            save_dict[f'{n}={b:.3f}'] = {
                'success_rate': torch.empty(0, dtype=torch.float32, device=device),
                'adversarial_samples': {}
            }

    print('[INFO] Computing attacks on batches...')
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader)):

        # if batch_idx == 1:
        #     break

        # foolbox wants ep tensors for working faster
        images, labels = ep.astensors(*(images.to(device), labels.to(device)))

        for (name, attack) in zip(attack_names, attacks):

            if attack_type == 'blackbox':
                with torch.no_grad():
                    _, adv, success = attack(attacked_model, images, labels, epsilons=bounds)
            else:
                _, adv, success = attack(attacked_model, images, labels, epsilons=bounds)

            for i, b in enumerate(bounds):

                # save success rate of batch
                save_dict[f'{name}={b:.3f}']['success_rate'] = torch.cat(
                    [save_dict[f'{name}={b:.3f}']['success_rate'],
                     success[i].raw.to(torch.float32)]
                )

                # save each adversarial
                for img_idx in range(batch_size):

                    sample_n = f'{int(batch_idx * batch_size) + img_idx}'
                    img = adv[i][img_idx].raw.permute(1, 2, 0).cpu().numpy()
                    save_dict[f'{name}={b:.3f}']['adversarial_samples'][sample_n] = img

    # compute final success rate for each bound and save .pickle files
    for attack in save_dict.keys():
        file_name = f'{out_folder}{attack}.pickle'
        values_dict = save_dict[attack]
        values_dict['success_rate'] = values_dict['success_rate'].mean().item()
        with open(file_name, 'wb') as f:
            pickle.dump(values_dict, f)


if __name__ == '__main__':

    arguments = parse_args()
    main(
        attack_type=arguments.attacks_type,
        images_path=arguments.images_path,
        batch_size=arguments.batch_size,
        attacked_net_path=arguments.attacked_net_path,
        attacked_net_type=arguments.attacked_net_type,
        out_folder=arguments.results_folder
    )