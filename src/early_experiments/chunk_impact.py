"""
. Plot some reconstructions with added noise on chunks
. measure L2 and Linf differences
"""

import torch
from torchvision.utils import make_grid
from torchmetrics import Accuracy
from einops import rearrange, pack
from kornia.enhance import normalize

from robustbench import load_cifar10

import os
from tqdm import tqdm
from matplotlib import pyplot as plt

from src.NVAE.original.model import AutoEncoder
from src.NVAE.original.utils import get_arch_cells, pre_process


def main():
    # load nvae pretrained cifar10
    checkpoint = torch.load(CKPT_NVAE, map_location='cpu')

    # get and update args
    args = checkpoint['args']
    args.num_mixture_dec = 10
    args.batch_size = 100

    # init model and load
    arch_instance = get_arch_cells(args.arch_instance)
    nvae = AutoEncoder(args, None, arch_instance)
    nvae.load_state_dict(checkpoint['state_dict'], strict=False)
    nvae = nvae.cuda().eval()

    # load cifar10
    x_test, y_test = load_cifar10(n_examples=200, data_dir=ADV_BASE_PATH)
    data_n = x_test.shape[0]
    y_test = y_test.cuda()

    # load classifiers pretrained cifar10
    os.environ["TORCH_HOME"] = TORCH_HOME
    resnet32 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True)
    resnet32 = resnet32.cuda().eval()
    vgg16_bn = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
    vgg16_bn = vgg16_bn.cuda().eval()

    # measure accuracy.
    accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1).cuda()

    for sigma in [0.8, 1.2, 1.6, 2.0]:

        # for each batch:
        # reconstruct perturbing all 30 chunks (also case with no perturbations)
        # try different values for sigma param
        params = (0., sigma, 1.)

        # on each image, compute L2 and Linf losses w.r.t original image.
        all_l2 = torch.empty((0, 32), device='cuda:0')
        all_li = torch.empty((0, 32), device='cuda:0')
        all_images = torch.empty((0, 32, 3, 32, 32), device='cuda:0')  # 0, n_chunks + rec + orig, c, h, w

        for b in tqdm(range(0, data_n, args.batch_size)):

            batch_x = x_test[b:b + args.batch_size].cuda()
            recons_batch = batch_x.unsqueeze(1)

            # init losses
            l2_batch = torch.zeros((args.batch_size, 1), device='cuda:0')
            li_batch = torch.zeros((args.batch_size, 1), device='cuda:0')

            for chunk in range(-1, 30):
                # reconstruct cifar10 test set
                with torch.no_grad():
                    logits = nvae.my_forward(pre_process(batch_x, args.num_x_bits), chunk=chunk, params=params)
                    decoder = nvae.decoder_output(logits)
                    this_recons = decoder.sample()

                recons_batch, _ = pack([recons_batch, this_recons.unsqueeze(1)], 'b * c h w')

                # get errors
                this_l2 = torch.mean(torch.cdist(batch_x, this_recons).view(args.batch_size, -1), dim=1).unsqueeze(1)
                l2_batch, _ = pack([l2_batch, this_l2], 'b *')

                this_li = torch.mean(torch.cdist(batch_x, this_recons, p=float('inf')).view(args.batch_size, -1),
                                     dim=1).unsqueeze(1)
                li_batch, _ = pack([li_batch, this_li], 'b *')

            # add to all
            all_l2, _ = pack([all_l2, l2_batch], '* n')
            all_li, _ = pack([all_li, li_batch], '* n')
            all_images, _ = pack([all_images, recons_batch], '* n c h w')

            # plot
            if b == 0:
                imgs = rearrange(all_images[:20], 'b n c h w -> (b n) c h w')
                imgs = make_grid(imgs, nrow=32).permute(1, 2, 0).cpu().numpy()
                plt.imshow(imgs)
                plt.axis(False)
                plt.title(f'Single Chunks modification (CIFAR 10 Test set) - sigma = {sigma}')
                plt.show()

        # mean error per chunk: (max l2 = 0.5 - max linf = 8/255 = 0.03137
        l2_mean_per_chunk = torch.mean(all_l2, dim=0).cpu().numpy()
        li_mean_per_chunk = torch.mean(all_li, dim=0).cpu().numpy()
        del all_l2
        del all_li

        print(f'mean L2 error on original, recons and single chunk perturbations: {l2_mean_per_chunk}')
        print(f'mean Linf error on original, recons and single chunk perturbations: {li_mean_per_chunk}')

        fig, (l, r) = plt.subplots(1, 2, figsize=(12, 8))
        l.plot(l2_mean_per_chunk[2:])
        l.set_title('mean L2 per chunk perturbation')
        r.plot(li_mean_per_chunk[2:])
        r.set_title('mean Linf per chunk perturbation')
        fig.suptitle(f'SIGMA = {sigma}')
        plt.show()

        # measure accuracies
        all_images = rearrange(all_images, 'b n c h w -> n b c h w')

        for chunk in range(all_images.shape[0]):

            this_images = normalize(all_images[chunk],
                                    mean=torch.tensor([0.507, 0.4865, 0.4409], device='cuda:0'),
                                    std=torch.tensor([0.2673, 0.2564, 0.2761], device='cuda:0'))

            with torch.no_grad():
                resnet_preds = torch.argmax(resnet32(this_images), dim=1)
                vgg16_preds = torch.argmax(vgg16_bn(this_images), dim=1)

                if chunk == 0:
                    print(f"Resnet32 clean acc: {accuracy(resnet_preds, y_test)}")
                    print(f"VGG16 clean acc: {accuracy(vgg16_preds, y_test)}")

                elif chunk == 1:
                    print(f"Resnet32 recons acc: {accuracy(resnet_preds, y_test)}")
                    print(f"VGG16 recons acc: {accuracy(vgg16_preds, y_test)}")

                else:
                    print(f"Resnet32 chunk perturbed = {chunk - 2} acc: {accuracy(resnet_preds, y_test)}")
                    print(f"VGG16 chunk perturbed = {chunk - 2} acc: {accuracy(vgg16_preds, y_test)}")


if __name__ == '__main__':
    CKPT_NVAE = '/media/dserez/runs/NVAE/cifar10/original.pt'
    ADV_BASE_PATH = '/media/dserez/code/adversarial/'
    TORCH_HOME = '/media/dserez/runs/classification/cifar10/'
    main()
