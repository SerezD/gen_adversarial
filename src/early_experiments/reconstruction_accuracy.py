"""
check if reconstruction accuracy drops (is it an attack already?).
"""
import torch
from torchvision.utils import make_grid
from torchmetrics import Accuracy
from einops import rearrange, pack
from kornia.enhance import normalize

from robustbench import load_cifar10

import os
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
    x_test, y_test = load_cifar10(data_dir=ADV_BASE_PATH)
    data_n = x_test.shape[0]

    # load classifiers pretrained cifar10
    os.environ["TORCH_HOME"] = TORCH_HOME
    resnet32 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True)
    resnet32 = resnet32.cuda().eval()
    vgg16_bn = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
    vgg16_bn = vgg16_bn.cuda().eval()

    accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1).cuda()

    with torch.no_grad():

        recons = torch.empty((0, 3, 32, 32), device='cuda:0')

        for b in range(0, data_n, args.batch_size):

            batch_x, _ = x_test[b:b + args.batch_size].cuda(), y_test[b:b + args.batch_size].cuda()

            # reconstruct cifar10 test set
            decoder = nvae.decoder_output(nvae(pre_process(batch_x, args.num_x_bits))[0])
            batch_recons = decoder.sample()
            recons, _ = pack([recons, batch_recons], '* c h w')

            # plot
            if b == 0:
                imgs = rearrange(torch.stack([batch_x, batch_recons], dim=0), 'n b c h w -> (b n) c h w')
                imgs = make_grid(imgs, nrow=20).permute(1, 2, 0).cpu().numpy()
                plt.imshow(imgs)
                plt.axis(False)
                plt.title('Reconstructions (CIFAR 10 Test set)')
                # plt.show()

        del nvae

        # measure accuracy.
        x_test = normalize(x_test.cuda(),
                           mean=torch.tensor([0.507, 0.4865, 0.4409], device='cuda:0'),
                           std=torch.tensor([0.2673, 0.2564, 0.2761], device='cuda:0'))

        recons = normalize(recons,
                           mean=torch.tensor([0.507, 0.4865, 0.4409], device='cuda:0'),
                           std=torch.tensor([0.2673, 0.2564, 0.2761], device='cuda:0'))

        y_test = y_test.cuda()

        resnet_clean_preds = torch.argmax(resnet32(x_test), dim=1)
        resnet_recon_preds = torch.argmax(resnet32(recons), dim=1)

        print(f"Resnet32 clean acc: {accuracy(resnet_clean_preds, y_test)} - "
              f"recons acc: {accuracy(resnet_recon_preds, y_test)}")

        vgg16_clean_preds = torch.argmax(vgg16_bn(x_test), dim=1)
        vgg16_recon_preds = torch.argmax(vgg16_bn(recons), dim=1)

        print(f"VGG16 clean acc: {accuracy(vgg16_clean_preds, y_test)} - "
              f"recons acc: {accuracy(vgg16_recon_preds, y_test)}")


if __name__ == '__main__':

    """
    [ORIGINAL]
    Resnet32 clean acc: 0.9301999807357788 - recons acc: 0.92330002784729
    VGG16 clean acc: 0.9340000152587891 - recons acc: 0.9323999881744385
    
    [REPRODUCTION]
    Resnet32 clean acc: 0.9301999807357788 - recons acc: 0.9276999831199646
    VGG16 clean acc: 0.9340000152587891 - recons acc: 0.9344000220298767
    
    [OURS_BASE]
    Resnet32 clean acc: 0.9301999807357788 - recons acc: 0.9049999713897705
    VGG16 clean acc: 0.9340000152587891 - recons acc: 0.9218999743461609

    [OURS_WEIGHTED]
    Resnet32 clean acc: 0.9301999807357788 - recons acc: 0.9039000272750854
    VGG16 clean acc: 0.9340000152587891 - recons acc: 0.9218000173568726
    """

    CKPT_NVAE = '/media/dserez/runs/NVAE/cifar10/ours_weighted.pt'
    ADV_BASE_PATH = '/media/dserez/code/adversarial/'
    TORCH_HOME = '/media/dserez/runs/classification/cifar10/'
    main()
