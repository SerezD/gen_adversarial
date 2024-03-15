import argparse
import json
import os
import torch
from einops import pack, rearrange
import foolbox as fb
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from data.datasets import ImageLabelDataset
from src.defenses.models import Cifar10ResnetModel, CelebAResnetModel, Cifar10VGGModel, Cifar10NVAEDefenseModel, \
    CelebAStyleGanDefenseModel


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def parse_args():

    """
    RUNS TO TRY

    --images_path '/media/dserez/datasets/cifar10/validation/' --batch_size 50
    --classifier_path '/media/dserez/runs/adversarial/CNNs/hub/cifar10_resnet32-ef93fc4d.pt' --classifier_type 'resnet-32'
    --autoencoder_path '/media/dserez/runs/NVAE/cifar10/ours/replica/large_latents.pt' --autoencoder_name 'NVAE_3'
    --resample_from '2'
    --results_folder './tmp/'

    --images_path '/media/dserez/datasets/cifar10/validation/' --batch_size 50
    --classifier_path '/media/dserez/runs/adversarial/CNNs/hub/cifar10_resnet32-ef93fc4d.pt' --classifier_type 'resnet-32'
    --autoencoder_path '/media/dserez/runs/NVAE/cifar10/ours/3scales_4groups.pt' --autoencoder_name 'NVAE_3x4'
    --resample_from '8'
    --results_folder './tmp/'

    --images_path '/media/dserez/datasets/cifar10/validation/' --batch_size 50
    --classifier_path '/media/dserez/runs/adversarial/CNNs/hub/cifar10_vgg16_bn-6ee7ea24.pt' --classifier_type 'vgg-16'
    --autoencoder_path '/media/dserez/runs/NVAE/cifar10/ours/replica/large_latents.pt' --autoencoder_name 'NVAE_3'
    --resample_from '2'
    --results_folder './tmp/'

    --images_path '/media/dserez/datasets/cifar10/validation/' --batch_size 50
    --classifier_path '/media/dserez/runs/adversarial/CNNs/hub/cifar10_vgg16_bn-6ee7ea24.pt' --classifier_type 'vgg-16'
    --autoencoder_path '/media/dserez/runs/NVAE/cifar10/ours/3scales_4groups.pt' --autoencoder_name 'NVAE_3x4'
    --resample_from '8'
    --results_folder './tmp/'

    --images_path '/media/dserez/datasets/celeba_hq_gender/test/' --batch_size 25
    --classifier_path '/media/dserez/runs/adversarial/CNNs/resnet50_celeba_hq_gender/best.pt' --classifier_type 'resnet-50'
    --autoencoder_path '/media/dserez/runs/stylegan2/inversions/e4e_ffhq_encoder.pt' --autoencoder_name 'E4E_StyleGAN'
    --resample_from '9'
    --results_folder './tmp/'
    """

    parser = argparse.ArgumentParser('Test the defense model with some sanity checks')

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

    args = parser.parse_args()

    # create folder
    args.results_folder = f'{args.results_folder}/{args.autoencoder_name}_{args.classifier_type}/'
    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    args.plots_folder = f'{args.results_folder}/plots_brute_force/'
    if not os.path.exists(args.plots_folder):
        os.makedirs(args.plots_folder)

    return args


@torch.no_grad()
def main(args: argparse.Namespace):
    """
    Given a CNN model + Defense, test:
    - clean accuracy of CNN and CNN + Defense.
    - accuracies degradation when adding increasingly larger gaussian noise (std).
    - simple brute-force attack with random noise of the correct norm (best of N attempts, for different norms).
    """

    # load pre-trained models
    if args.classifier_type == 'resnet-32':
        base_classifier = Cifar10ResnetModel(args.classifier_path, device)
        defense_model = Cifar10NVAEDefenseModel(base_classifier, args.autoencoder_path, device, args.resample_from)
        args.image_size = 32
        std_params = (0.05, 0.10, 0.15, 0.20, 0.25)
        bounds_l2 = (0.1, 0.5, 1.0)
    elif args.classifier_type == 'vgg-16':
        base_classifier = Cifar10VGGModel(args.classifier_path, device)
        defense_model = Cifar10NVAEDefenseModel(base_classifier, args.autoencoder_path, device, args.resample_from)
        args.image_size = 32
        std_params = (0.05, 0.10, 0.15, 0.20, 0.25)
        bounds_l2 = (0.1, 0.5, 1.0)
    elif args.classifier_type == 'resnet-50':
        base_classifier = CelebAResnetModel(args.classifier_path, device)
        defense_model = CelebAStyleGanDefenseModel(base_classifier, args.autoencoder_path, device, args.resample_from)
        args.image_size = 256
        std_params = (0.10, 0.20, 0.30, 0.40, 0.50)
        bounds_l2 = (0.1, 0.5, 1.0)
    else:
        raise ValueError(f'Unknown classifier type: {args.classifier_type}')

    defense_model = defense_model.eval()

    # dataloader
    dataloader = DataLoader(ImageLabelDataset(folder=args.images_path, image_size=args.image_size),
                            batch_size=args.batch_size, shuffle=False)

    # params for measuring clean accuracy
    gt_labels = torch.empty((0, 1), device=device)
    base_cl_preds_on_clean = torch.empty((0, 1), device=device)
    defense_preds_on_clean = torch.empty((0, 1), device=device)

    # params for measuring random gaussian noise
    n = len(std_params)
    base_cl_preds_on_gauss = torch.empty((n, 0, 1), device=device)
    defense_preds_on_gauss = torch.empty((n, 0, 1), device=device)

    # params for brute force L2 attack
    def_attacked_model = fb.PyTorchModel(defense_model, bounds=(0, 1), device=device)

    if base_classifier.preprocessing is not None:
        preprocessing = dict(mean=base_classifier.mean, std=base_classifier.std, axis=-3)
    else:
        preprocessing = None

    base_attacked_model = fb.PyTorchModel(base_classifier.classifier, bounds=(0, 1), device=device,
                                          preprocessing=preprocessing)

    attack = fb.attacks.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack(repeats=1000)  # TODO CHECK this number!

    base_success_rate = torch.empty((len(bounds_l2), 0), device=device)
    def_success_rate = torch.empty((len(bounds_l2), 0), device=device)

    for b_idx, (images, labels) in enumerate(dataloader):

        # if b_idx == 2:  # TODO REMOVE
        #     break

        images = images.to(device)
        labels = labels.to(device)
        gt_labels, _ = pack([gt_labels, labels.unsqueeze(1)], '* d')

        # 1 clean accuracy
        batch_base_cl_preds_on_clean, batch_defense_preds_on_clean, _ = defense_model(images, preds_only=False)
        batch_base_cl_preds_on_clean = torch.argmax(batch_base_cl_preds_on_clean, dim=1, keepdim=True)
        batch_defense_preds_on_clean = torch.argmax(batch_defense_preds_on_clean, dim=1, keepdim=True)

        base_cl_preds_on_clean, _ = pack([base_cl_preds_on_clean, batch_base_cl_preds_on_clean], '* d')
        defense_preds_on_clean, _ = pack([defense_preds_on_clean, batch_defense_preds_on_clean], '* d')

        # 2 increasing gaussian noise
        stds = torch.ones_like(images).repeat(n, 1, 1, 1, 1)
        for i in range(n):
            stds[i] *= std_params[i]
        gaussian_noise = torch.normal(mean=0., std=stds)
        perturbed_images = torch.clip(images.repeat(n, 1, 1, 1, 1) + gaussian_noise, 0.0, 1.0)
        perturbed_images = rearrange(perturbed_images, 'n b c h w -> (n b) c h w')

        batch_base_cl_preds_on_gauss, batch_defense_preds_on_gauss, _ = defense_model(perturbed_images, preds_only=False)
        batch_base_cl_preds_on_gauss = torch.argmax(batch_base_cl_preds_on_gauss, dim=1, keepdim=True)
        batch_base_cl_preds_on_gauss = rearrange(batch_base_cl_preds_on_gauss, '(n b) d -> n b d', n=n)
        batch_defense_preds_on_gauss = torch.argmax(batch_defense_preds_on_gauss, dim=1, keepdim=True)
        batch_defense_preds_on_gauss = rearrange(batch_defense_preds_on_gauss, '(n b) d -> n b d', n=n)

        base_cl_preds_on_gauss, _ = pack([base_cl_preds_on_gauss, batch_base_cl_preds_on_gauss], 'n * d')
        defense_preds_on_gauss, _ = pack([defense_preds_on_gauss, batch_defense_preds_on_gauss], 'n * d')

        # 3 Brute Force Attack with increasing L2 bounds.
        # adv = list of len(bounds_l2) of tensors B C H W.
        # success = boolean tensor of shape (N_bounds, B)
        _, _, suc_base = attack(base_attacked_model, images, labels.to(device), epsilons=bounds_l2)
        _, adv, suc_def = attack(def_attacked_model, images, labels.to(device), epsilons=bounds_l2)

        # cat results
        base_success_rate, _ = pack([base_success_rate, suc_base], 'n *')
        def_success_rate, _ = pack([def_success_rate, suc_def], 'n *')

        # save visual examples of applied perturbation.
        if b_idx % 20 == 0:

            # estimate of the cleaned image (estimate because the process is random)
            cleaned_imgs = [defense_model(found_adv, preds_only=False)[-1] for found_adv in adv]

            for i, b in enumerate(bounds_l2):

                max_b = 8
                img_i = images.clone()[:max_b]
                adv_i = adv[i][:max_b]
                def_i = cleaned_imgs[i][:max_b]
                max_b = adv_i.shape[0]

                examples = torch.cat([img_i, adv_i, def_i], dim=0)
                display = make_grid(examples, nrow=max_b).permute(1, 2, 0).cpu()
                plt.imshow(display)
                plt.axis(False)
                plt.title(f'originals, adversarial and cleaned images at L2={b:.2f}')
                plt.savefig(f'{args.plots_folder}batch_{b_idx}_bound={b:.2f}.png')
                plt.close()

    # compute and save final results
    res_dict = {}

    cl_accuracy_on_clean = torch.eq(base_cl_preds_on_clean, gt_labels).to(torch.float32).mean().item()
    def_accuracy_on_clean = torch.eq(defense_preds_on_clean, gt_labels).to(torch.float32).mean().item()

    res_dict['clean accuracy base'] = cl_accuracy_on_clean
    res_dict['clean accuracy defense'] = def_accuracy_on_clean

    cl_accuracy_on_gauss = torch.eq(base_cl_preds_on_gauss, gt_labels).to(torch.float32).mean(dim=1)
    def_accuracy_on_gauss = torch.eq(defense_preds_on_gauss, gt_labels).to(torch.float32).mean(dim=1)

    for i in range(n):
        res_dict[f'gauss std={std_params[i]:.2f} accuracy base'] = cl_accuracy_on_gauss[i].item()
        res_dict[f'gauss std={std_params[i]:.2f} accuracy defense'] = def_accuracy_on_gauss[i].item()

    for i, b in enumerate(bounds_l2):
        res_dict[f'brute force l2 bound={b:.2f} success base'] = base_success_rate[i].mean().item()
        res_dict[f'brute force l2 bound={b:.2f} success defense'] = def_success_rate[i].mean().item()

    with open(f'{args.results_folder}results.json', 'w') as f:
        json.dump(res_dict, f)


if __name__ == '__main__':

    arguments = parse_args()
    main(arguments)

