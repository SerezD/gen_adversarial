"""
. compute and save reconstructions turning off each single chunk
. measure L2 Linf and SSIM
. measure accuracies on resnet32 and vgg16
"""

import torch
from torchmetrics import Accuracy
from torchmetrics.image import StructuralSimilarityIndexMeasure
from einops import pack
from kornia.enhance import normalize

from robustbench import load_cifar10

import os
from tqdm import tqdm
import numpy as np

from src.NVAE.original.model import AutoEncoder
from src.NVAE.original.utils import get_arch_cells, pre_process


def main():

    # load cifar10
    x_test, y_test = load_cifar10(data_dir=ADV_BASE_PATH)
    data_n = x_test.shape[0]
    x_test = x_test.cuda(DEV_0)

    # LOAD IMAGES IF ALREADY COMPUTED
    if os.path.exists(IMAGES_FILE):

        all_recons = np.load(IMAGES_FILE)
        all_recons = torch.tensor(all_recons)

    else:
        # go and compute!
        # load nvae pretrained cifar10
        checkpoint = torch.load(CKPT_NVAE, map_location='cpu')

        # get and update args
        args = checkpoint['args']
        args.num_mixture_dec = 10
        args.batch_size = BATCH_SIZE

        latents_per_group = args.num_latent_per_group
        initial_res = 32  # CIFAR 10
        scales = args.num_latent_scales
        groups_per_scale = [
                max(args.min_groups_per_scale, args.num_groups_per_scale // 2)
                if args.ada_groups else
                args.num_groups_per_scale
                for _ in range(args.num_latent_scales)
            ]  # different on each scale if is_adaptive

        # init model and load
        arch_instance = get_arch_cells(args.arch_instance)
        nvae = AutoEncoder(args, None, arch_instance)
        nvae.load_state_dict(checkpoint['state_dict'], strict=False)
        nvae = nvae.cuda(DEV_0).eval()

        # get all reconstructions and save them as single numpy vector
        all_recons = torch.empty((0, 1 + sum(groups_per_scale), 3, initial_res, initial_res), device=f'cuda:{DEV_0}')

        for b in tqdm(range(0, data_n, args.batch_size)):

            batch_final = torch.empty((args.batch_size, 0, 3, initial_res, initial_res), device=f'cuda:{DEV_0}')

            batch_x = x_test[b:b + args.batch_size].cuda(DEV_0)

            # get simple batch reconstruction
            with torch.no_grad():
                logits = nvae.forward(pre_process(batch_x, args.num_x_bits))[0]
                decoder = nvae.decoder_output(logits)
                simple_recons = decoder.sample()
            batch_final, _ = pack([batch_final, simple_recons.unsqueeze(1)], 'b * c h w')

            for s in range(scales):

                base_n = sum(groups_per_scale[:s])

                for g in range(groups_per_scale[s]):

                    with torch.no_grad():
                        logits = nvae.my_forward(pre_process(batch_x, args.num_x_bits), chunk=base_n + g)
                        decoder = nvae.decoder_output(logits)
                        chunk_recons = decoder.sample()
                    batch_final, _ = pack([batch_final, chunk_recons.unsqueeze(1)], 'b * c h w')

            all_recons, _ = pack([all_recons, batch_final], '* n c h w')

        # save
        np.save(IMAGES_FILE, all_recons.cpu().numpy())

    x_test = x_test.to(f'cuda:{DEV_1}')
    all_recons, _ = pack([x_test.unsqueeze(1), all_recons.to(f'cuda:{DEV_1}')], 'b * c h w')

    ssim = StructuralSimilarityIndexMeasure().to(f'cuda:{DEV_1}')

    # load classifiers pretrained cifar10
    os.environ["TORCH_HOME"] = TORCH_HOME
    resnet32 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True)
    resnet32 = resnet32.cuda(DEV_1).eval()
    vgg16_bn = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
    vgg16_bn = vgg16_bn.cuda(DEV_1).eval()

    # measure accuracy.
    accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1).cuda(DEV_1)
    y_test = y_test.cuda(DEV_1)

    errors = {}
    accuracies = {}

    for i in range(all_recons.shape[1]):

        if i > 0:
            l2_error = torch.mean(torch.cdist(x_test, all_recons[:, i]).view(data_n, -1))
            li_error = torch.mean(torch.cdist(x_test, all_recons[:, i], p=float('inf')).view(data_n, -1))
            ssim_error = ssim(all_recons[:, i], x_test)

            errors['recons' if i == 1 else f'chunk_{(all_recons.shape[1] - i) - 1}'] = {
                'L2': l2_error.item(),
                'Linf': li_error.item(),
                'ssim': ssim_error.item()
            }
        else:
            errors['clean'] = {
                'L2': 0.0,
                'Linf': 0.0,
                'ssim': 1.0
            }

        # measure accuracies

        normalized_imgs = normalize(all_recons[:, i],
                                    mean=torch.tensor([0.507, 0.4865, 0.4409], device='cuda:0'),
                                    std=torch.tensor([0.2673, 0.2564, 0.2761], device='cuda:0'))

        with torch.no_grad():

            # pred
            resnet_preds = torch.argmax(resnet32(normalized_imgs), dim=1)
            vgg16_preds = torch.argmax(vgg16_bn(normalized_imgs), dim=1)

        if i == 0:
            acc_key = 'clean'
        elif i == 1:
            acc_key = 'recons'
        else:
            acc_key = f'chunk_{(all_recons.shape[1] - i) - 1}'

        accuracies[acc_key] = {
            'resnet32': accuracy(resnet_preds, y_test).item(),
            'vgg16': accuracy(vgg16_preds, y_test).item()
        }

    print(errors)
    print(accuracies)


if __name__ == '__main__':

    # CKPT_NVAE = '/media/dserez/runs/NVAE/cifar10/best/ours_base.pt'
    # ADV_BASE_PATH = '/media/dserez/code/adversarial/'
    # TORCH_HOME = '/media/dserez/runs/classification/cifar10/'
    # IMAGES_FILE = f'{ADV_BASE_PATH}reconstructions_ours_base.npy'
    # DEV_0, DEV_1 = [int(i) for i in os.getenv("CUDA_VISIBLE_DEVICES").split(',')]

    CKPT_NVAE = '../media/ours_weighted.pt'
    ADV_BASE_PATH = '../media/'
    TORCH_HOME = '../media/cifar10/'
    IMAGES_FILE = f'{ADV_BASE_PATH}reconstructions_ours_weighted.npy'
    DEV_0, DEV_1 = 0, 0

    BATCH_SIZE = 25

    main()
