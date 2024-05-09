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

from data.datasets import ImageLabelDataset
from src.defenses.models import Cifar10ResnetModel, CelebAResnetModel, Cifar10VGGModel, Cifar10NVAEDefenseModel, \
    CelebAStyleGanDefenseModel


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def parse_args():

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
    - accuracies degradation when using negative images attack
    - simple brute-force attack with random noise of the correct norm (best of N attempts, for different norms).
    """

    # load pre-trained models
    if args.classifier_type == 'resnet-32':
        bounds_l2 = (0.5, 1.0, 2.0, 4.0)
        gaussian_eps = bounds_l2[-1]
        args.image_size = 32
        base_classifier = Cifar10ResnetModel(args.classifier_path, device)
        defense_model = Cifar10NVAEDefenseModel(base_classifier, args.autoencoder_path, device, args.resample_from,
                                                gaussian_eps=gaussian_eps)
    elif args.classifier_type == 'vgg-16':
        bounds_l2 = (0.5, 1.0, 2.0, 4.0)
        gaussian_eps = bounds_l2[-1]
        args.image_size = 32
        base_classifier = Cifar10VGGModel(args.classifier_path, device)
        defense_model = Cifar10NVAEDefenseModel(base_classifier, args.autoencoder_path, device, args.resample_from,
                                                gaussian_eps=gaussian_eps)
    elif args.classifier_type == 'resnet-50':
        bounds_l2 = (0.5, 1.0, 2.0, 4.0)
        gaussian_eps = bounds_l2[-1]
        args.image_size = 256
        base_classifier = CelebAResnetModel(args.classifier_path, device)
        defense_model = CelebAStyleGanDefenseModel(base_classifier, args.autoencoder_path, device, args.resample_from,
                                                   gaussian_eps=gaussian_eps)
    else:
        raise ValueError(f'Unknown classifier type: {args.classifier_type}')

    # dataloader
    dataloader = DataLoader(ImageLabelDataset(folder=args.images_path, image_size=args.image_size),
                            batch_size=args.batch_size, shuffle=False)

    # base classifier to measure undefended performances
    fb_base_model = fb.PyTorchModel(base_classifier.eval(), bounds=(0, 1), device=device)

    # wrap defense with EOT to maintain deterministic predictions
    # this is a simple counter_attack, to avoid gradient masking
    # TODO ensure to maintain the same n_steps in all cases
    fb_defense_model = fb.PyTorchModel(defense_model.eval(), bounds=(0, 1), device=device)
    fb_defense_model = fb.models.ExpectationOverTransformationWrapper(fb_defense_model, n_steps=8)

    # params for measuring clean accuracy
    gt_labels = torch.empty((0, 1), device=device)
    base_cl_preds_on_clean = torch.empty((0, 1), device=device)
    defense_preds_on_clean = torch.empty((0, 1), device=device)

    # params for L2 naive attacks
    bf_attack = fb.attacks.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack(repeats=1024)  # TODO CHECK this number!
    iv_attack = fb.attacks.InversionAttack(distance=fb.distances.l2)

    n_steps = 15  # one batch each x

    base_success_rate = torch.empty((2, len(bounds_l2), 0), device=device)
    def_success_rate = torch.empty((2, len(bounds_l2), 0), device=device)

    for b_idx, (images, labels) in enumerate(tqdm(dataloader)):

        # if b_idx == 2:  # TODO REMOVE
        #     break

        if b_idx % n_steps != 0:
            continue

        images = torch.clip(images.to(device), 0.0, 1.0)
        labels = labels.to(device)
        gt_labels, _ = pack([gt_labels, labels.unsqueeze(1)], '* d')

        # 1 clean accuracy
        batch_base_cl_preds_on_clean = fb_base_model(images)
        batch_defense_preds_on_clean = fb_defense_model(images)
        batch_base_cl_preds_on_clean = torch.argmax(batch_base_cl_preds_on_clean, dim=1, keepdim=True)
        batch_defense_preds_on_clean = torch.argmax(batch_defense_preds_on_clean, dim=1, keepdim=True)

        base_cl_preds_on_clean, _ = pack([base_cl_preds_on_clean, batch_base_cl_preds_on_clean], '* d')
        defense_preds_on_clean, _ = pack([defense_preds_on_clean, batch_defense_preds_on_clean], '* d')

        # # 2 Inversion Attack.
        # # adv = list of len(bounds_l2) of tensors B C H W.
        # # success = boolean tensor of shape (N_bounds, B)
        # _, _, suc_base_iv = iv_attack(fb_base_model, images, labels.to(device), epsilons=(0.5,))
        # _, _, suc_def_iv = iv_attack(fb_defense_model, images, labels.to(device), epsilons=(0.5,))
        #
        # # 3 Brute Force Attack with increasing L2 bounds.
        # # adv = list of len(bounds_l2) of tensors B C H W.
        # # success = boolean tensor of shape (N_bounds, B)
        # _, _, suc_base_bf = bf_attack(fb_base_model, images, labels.to(device), epsilons=bounds_l2)
        # _, adv, suc_def_bf = bf_attack(fb_defense_model, images, labels.to(device), epsilons=bounds_l2)
        #
        # # cat results
        # suc_base = torch.stack([suc_base_iv.repeat(len(bounds_l2), 1), suc_base_bf], dim=0)
        # suc_def = torch.stack([suc_def_iv.repeat(len(bounds_l2), 1), suc_def_bf], dim=0)
        # base_success_rate, _ = pack([base_success_rate, suc_base], 'd n *')
        # def_success_rate, _ = pack([def_success_rate, suc_def], 'd n *')

        # # save visual examples of applied perturbation.
        # if b_idx % 20 == 0:
        #
        #     # estimate of the cleaned image (estimate because the process is random)
        #     cleaned_imgs = [defense_model(found_adv, preds_only=False)[-1] for found_adv in adv]
        #
        #     for i, b in enumerate(bounds_l2):
        #
        #         max_b = 8
        #         img_i = images.clone()[:max_b]
        #         adv_i = adv[i][:max_b]
        #         def_i = cleaned_imgs[i][:max_b]
        #         max_b = adv_i.shape[0]
        #
        #         examples = torch.cat([img_i, adv_i, def_i], dim=0)
        #         display = make_grid(examples, nrow=max_b).permute(1, 2, 0).cpu()
        #         plt.imshow(display)
        #         plt.axis(False)
        #         plt.title(f'originals, adversarial and cleaned images at L2={b:.2f}')
        #         plt.savefig(f'{args.plots_folder}batch_{b_idx}_bound={b:.2f}.png')
        #         plt.close()

    # compute and save final results
    res_dict = {}

    cl_accuracy_on_clean = torch.eq(base_cl_preds_on_clean, gt_labels).to(torch.float32).mean().item()
    def_accuracy_on_clean = torch.eq(defense_preds_on_clean, gt_labels).to(torch.float32).mean().item()

    print(f'clean accuracy base: {cl_accuracy_on_clean}')
    print(f'clean accuracy defense: {def_accuracy_on_clean}')

    # res_dict['clean accuracy base'] = cl_accuracy_on_clean
    # res_dict['clean accuracy defense'] = def_accuracy_on_clean
    #
    # res_dict[f'inversion attack success base'] = base_success_rate[0, 0].mean().item()
    # res_dict[f'inversion attack success defense'] = def_success_rate[0, 0].mean().item()
    #
    # for i, b in enumerate(bounds_l2):
    #     # print(f'brute force l2 bound={b:.2f} success base = {base_success_rate[1, i].mean().item()}')
    #     res_dict[f'brute force l2 bound={b:.2f} success base'] = base_success_rate[1, i].mean().item()
    #     res_dict[f'brute force l2 bound={b:.2f} success defense'] = def_success_rate[1, i].mean().item()
    #
    # with open(f'{args.results_folder}results.json', 'w') as f:
    #     json.dump(res_dict, f)


if __name__ == '__main__':

    arguments = parse_args()
    main(arguments)

