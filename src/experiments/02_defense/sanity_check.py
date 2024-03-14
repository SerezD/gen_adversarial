import argparse
import os
import torch
from einops import pack, rearrange
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
    --autoencoder_path '/media/dserez/runs/NVAE/cifar10/ours/replica/large_latents.pt' --resample_from '2'
    --results_folder './tmp/'

    --images_path '/media/dserez/datasets/cifar10/validation/' --batch_size 50
    --classifier_path '/media/dserez/runs/adversarial/CNNs/hub/cifar10_resnet32-ef93fc4d.pt' --classifier_type 'resnet-32'
    --autoencoder_path '/media/dserez/runs/NVAE/cifar10/ours/3scales_4groups.pt' --resample_from '8'
    --results_folder './tmp/'

    --images_path '/media/dserez/datasets/cifar10/validation/' --batch_size 50
    --classifier_path '/media/dserez/runs/adversarial/CNNs/hub/cifar10_vgg16_bn-6ee7ea24.pt' --classifier_type 'vgg-16'
    --autoencoder_path '/media/dserez/runs/NVAE/cifar10/ours/replica/large_latents.pt' --resample_from '2'
    --results_folder './tmp/'

    --images_path '/media/dserez/datasets/cifar10/validation/' --batch_size 50
    --classifier_path '/media/dserez/runs/adversarial/CNNs/hub/cifar10_vgg16_bn-6ee7ea24.pt' --classifier_type 'vgg-16'
    --autoencoder_path '/media/dserez/runs/NVAE/cifar10/ours/3scales_4groups.pt' --resample_from '8'
    --results_folder './tmp/'

    --images_path '/media/dserez/datasets/celeba_hq_gender/test/' --batch_size 25
    --classifier_path '/media/dserez/runs/adversarial/CNNs/resnet50_celeba_hq_gender/best.pt' --classifier_type 'resnet-50'
    --autoencoder_path '/media/dserez/runs/stylegan2/inversions/e4e_ffhq_encoder.pt' --resample_from '9'
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

    parser.add_argument('--resample_from', type=int, required=True,
                        help='hierarchy level where re-sampling defense starts')

    parser.add_argument('--results_folder', type=str, required=True,
                        help='folder to save .pickle file with results')

    args = parser.parse_args()

    # create folder
    args.results_folder = f'{args.results_folder}/'
    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

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
    elif args.classifier_type == 'vgg-16':
        base_classifier = Cifar10VGGModel(args.classifier_path, device)
        defense_model = Cifar10NVAEDefenseModel(base_classifier, args.autoencoder_path, device, args.resample_from)
        args.image_size = 32
    elif args.classifier_type == 'resnet-50':
        base_classifier = CelebAResnetModel(args.classifier_path, device)
        defense_model = CelebAStyleGanDefenseModel(base_classifier, args.autoencoder_path, device, args.resample_from)
        args.image_size = 256
    else:
        raise ValueError(f'Unknown classifier type: {args.classifier_type}')

    # dataloader
    dataloader = DataLoader(ImageLabelDataset(folder=args.images_path, image_size=args.image_size),
                            batch_size=args.batch_size, shuffle=False)

    gt_labels = torch.empty((0, 1), device='cpu')
    base_cl_preds_on_clean = torch.empty((0, 1), device='cpu')
    defense_preds_on_clean = torch.empty((0, 1), device='cpu')

    std_params = (0.04, 0.08, 0.12, 0.16, 0.20)
    n = len(std_params)
    base_cl_preds_on_gauss = torch.empty((n, 0, 1), device='cpu')
    defense_preds_on_gauss = torch.empty((n, 0, 1), device='cpu')

    for b_idx, (images, labels) in enumerate(dataloader):

        if b_idx == 40:
            break

        images = images.to(device)
        gt_labels, _ = pack([gt_labels, labels.unsqueeze(1)], '* d')

        # 1 clean accuracy
        batch_base_cl_preds_on_clean, batch_defense_preds_on_clean = defense_model(images)
        batch_base_cl_preds_on_clean = torch.argmax(batch_base_cl_preds_on_clean, dim=1, keepdim=True).cpu()
        batch_defense_preds_on_clean = torch.argmax(batch_defense_preds_on_clean, dim=1, keepdim=True).cpu()

        base_cl_preds_on_clean, _ = pack([base_cl_preds_on_clean, batch_base_cl_preds_on_clean], '* d')
        defense_preds_on_clean, _ = pack([defense_preds_on_clean, batch_defense_preds_on_clean], '* d')

        # 2 increasing gaussian noise
        stds = torch.ones_like(images).repeat(n, 1, 1, 1, 1)
        for i in range(n):
            stds[i] *= std_params[i]
        gaussian_noise = torch.normal(mean=0., std=stds)
        perturbed_images = torch.clip(images.repeat(n, 1, 1, 1, 1) + gaussian_noise, 0.0, 1.0)

        # visual example TODO -> decide if saving it or not
        # examples = perturbed_images[:, -1]
        # display = make_grid(examples, nrow=n).permute(1, 2, 0).cpu()
        # plt.imshow(display)
        # plt.axis(False)
        # plt.show()

        perturbed_images = rearrange(perturbed_images, 'n b c h w -> (n b) c h w')
        batch_base_cl_preds_on_gauss, batch_defense_preds_on_gauss = defense_model(perturbed_images)
        batch_base_cl_preds_on_gauss = torch.argmax(batch_base_cl_preds_on_gauss, dim=1, keepdim=True).cpu()
        batch_base_cl_preds_on_gauss = rearrange(batch_base_cl_preds_on_gauss, '(n b) d -> n b d', n=n)
        batch_defense_preds_on_gauss = torch.argmax(batch_defense_preds_on_gauss, dim=1, keepdim=True).cpu()
        batch_defense_preds_on_gauss = rearrange(batch_defense_preds_on_gauss, '(n b) d -> n b d', n=n)

        base_cl_preds_on_gauss, _ = pack([base_cl_preds_on_gauss, batch_base_cl_preds_on_gauss], 'n * d')
        defense_preds_on_gauss, _ = pack([defense_preds_on_gauss, batch_defense_preds_on_gauss], 'n * d')

    # compute final results
    cl_accuracy_on_clean = torch.eq(base_cl_preds_on_clean, gt_labels).to(torch.float32).mean().item()
    def_accuracy_on_clean = torch.eq(defense_preds_on_clean, gt_labels).to(torch.float32).mean().item()

    print('cl_accuracy_on_clean', cl_accuracy_on_clean)
    print('def_accuracy_on_clean', def_accuracy_on_clean)

    cl_accuracy_on_gauss = torch.eq(base_cl_preds_on_gauss, gt_labels).to(torch.float32).mean(dim=1)
    def_accuracy_on_gauss = torch.eq(defense_preds_on_gauss, gt_labels).to(torch.float32).mean(dim=1)

    for i in range(n):
        print('STD', std_params[i])
        print('cl_accuracy_on_gauss', cl_accuracy_on_gauss[i].item())
        print('def_accuracy_on_gauss', def_accuracy_on_gauss[i].item())
        print('#' * 30)


if __name__ == '__main__':

    arguments = parse_args()
    main(arguments)

