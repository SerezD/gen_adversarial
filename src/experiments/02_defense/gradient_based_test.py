import argparse
import json
import os
import torch
from einops import pack
import foolbox as fb
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.utils import make_grid
from tqdm import tqdm

from data.datasets import CoupledDataset
from src.defenses.models import Cifar10ResnetModel, CelebAResnetModel, Cifar10VGGModel, Cifar10NVAEDefenseModel, \
    CelebAStyleGanDefenseModel


def parse_args():

    """
    STYLEGAN
    --images_path
    /media/dserez/datasets/celeba_hq_gender/test/
    --batch_size
    1
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

    NVAE
    --images_path
    /media/dserez/datasets/cifar10/validation/
    --batch_size
    2
    --classifier_path
    /media/dserez/runs/adversarial/CNNs/hub/checkpoints/cifar10_resnet32-ef93fc4d.pt
    --classifier_type
    resnet-32
    --autoencoder_path
    /media/dserez/runs/NVAE/cifar10/ours/3scales_4groups.pt
    --autoencoder_name
    NVAE4x3
    --resample_from
    8
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


def main(rank: int, w_size: int, args: argparse.Namespace):
    """

    """

    device = f'cuda:{rank}'
    torch.cuda.set_device(rank)

    if 'MASTER_ADDR' not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"

    if 'MASTER_PORT' not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(backend='nccl', world_size=w_size, rank=rank)

    # load pre-trained models
    if args.classifier_type == 'resnet-32':
        base_classifier = Cifar10ResnetModel(args.classifier_path, device)
        defense_model = Cifar10NVAEDefenseModel(base_classifier, args.autoencoder_path, device, args.resample_from)
        args.image_size = 32
        bounds_l2 = (0.25, 0.5, 1.0, 2.0)
        n_steps = 8
    elif args.classifier_type == 'vgg-16':
        base_classifier = Cifar10VGGModel(args.classifier_path, device)
        defense_model = Cifar10NVAEDefenseModel(base_classifier, args.autoencoder_path, device, args.resample_from)
        args.image_size = 32
        bounds_l2 = (0.25, 0.5, 1.0, 2.0)
        n_steps = 8
    elif args.classifier_type == 'resnet-50':
        base_classifier = CelebAResnetModel(args.classifier_path, device)
        defense_model = CelebAStyleGanDefenseModel(base_classifier, args.autoencoder_path, device, args.resample_from)
        args.image_size = 256
        bounds_l2 = (0.25, 0.5, 1.0, 2.0)
        n_steps = 8
    else:
        raise ValueError(f'Unknown classifier type: {args.classifier_type}')

    base_model = base_classifier.classifier.eval()
    defense_model = defense_model.eval()

    # dataloader
    dataset = CoupledDataset(folder=args.images_path, image_size=args.image_size)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler)

    # create FoolBox Models (move preprocessing phase before attack)
    if defense_model.preprocess:
        preprocessing = dict(mean=defense_model.mean, std=defense_model.std, axis=-3)
        defense_model.preprocess = False
    else:
        preprocessing = None

    # wrap defense with EOT to maintain deterministic predictions
    fb_defense_model = fb.PyTorchModel(defense_model, bounds=(0, 1), device=device,
                                       preprocessing=preprocessing)
    fb_defense_model = fb.models.ExpectationOverTransformationWrapper(fb_defense_model, n_steps=n_steps)

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
        if rank == 0 and b_idx % 20 == 0:

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

        # time of attack greatly depends on input images, need this to avoid timeout errors
        dist.barrier()

    dist.all_reduce(base_success_rate, op=dist.ReduceOp.SUM)
    base_success_rate = base_success_rate / w_size

    dist.all_reduce(def_success_rate, op=dist.ReduceOp.SUM)
    def_success_rate = def_success_rate / w_size

    # compute and save final results
    if rank == 0:
        res_dict = {}

        for i, b in enumerate(bounds_l2):
            # print(f'brute force l2 bound={b:.2f} success base {base_success_rate[i].mean().item()}')
            # print(f'brute force l2 bound={b:.2f} success defense {def_success_rate[i].mean().item()}')

            res_dict[f'l2 bound={b:.2f} success on base'] = base_success_rate[i].mean().item()
            res_dict[f'l2 bound={b:.2f} success on defense'] = def_success_rate[i].mean().item()

        print('Opening file...')
        with open(f'{args.results_folder}results_{args.attack}_resample:{args.resample_from}.json', 'w') as f:
             json.dump(res_dict, f)

    dist.destroy_process_group()


if __name__ == '__main__':

    arguments = parse_args()
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(main, nprocs=WORLD_SIZE, args=(WORLD_SIZE, arguments))

