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
from tqdm import tqdm

from src.NVAE.mine.distributions import DiscMixLogistic
from src.NVAE.mine.model import AutoEncoder as NVAE
from src.NVAE.original.model import AutoEncoder
from src.NVAE.original.utils import get_arch_cells, pre_process


def get_model_conf(filepath: str):
    import yaml

    # load params
    with open(filepath, 'r', encoding='utf-8') as stream:
        params = yaml.safe_load(stream)

    return params


def main():

    if IS_OURS:

        conf_file = '/'.join(CKPT_NVAE.split('/')[:-1]) + '/conf.yaml'
        config = get_model_conf(conf_file)

        # create model and move it to GPU with id rank
        nvae = NVAE(config['autoencoder'], config['resolution'])

        checkpoint = torch.load(CKPT_NVAE, map_location='cpu')
        nvae.load_state_dict(checkpoint['state_dict'])
        nvae.cuda().eval()

    else:

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
    batch_size = 100

    # load classifiers pretrained cifar10
    os.environ["TORCH_HOME"] = TORCH_HOME
    resnet32 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True)
    resnet32 = resnet32.cuda().eval()
    vgg16_bn = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
    vgg16_bn = vgg16_bn.cuda().eval()

    accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1).cuda()

    with torch.no_grad():

        recons = torch.empty((0, 3, 32, 32), device='cuda:0')

        for b in tqdm(range(0, data_n, batch_size)):

            batch_x, _ = x_test[b:b + batch_size].cuda(), y_test[b:b + batch_size].cuda()

            # reconstruct cifar10 test set
            if IS_OURS:
                logits = nvae.autoencode(batch_x)
                batch_recons = DiscMixLogistic(logits, num_bits=8).mean()
            else:
                logits = nvae.decode(nvae.encode_deterministic(batch_x))
                decoder = nvae.decoder_output(logits)
                batch_recons = decoder.mean()

            recons, _ = pack([recons, batch_recons], '* c h w')

            # plot
            # if b == 0:
            #     imgs = rearrange(torch.stack([batch_x, batch_recons], dim=0), 'n b c h w -> (b n) c h w')
            #     imgs = make_grid(imgs, nrow=20).permute(1, 2, 0).cpu().numpy()
            #     plt.imshow(imgs)
            #     plt.axis(False)
            #     plt.title('Reconstructions (CIFAR 10 Test set)')
                # plt.show()

        del nvae

        # L2 error:
        l2 = torch.mean(torch.cdist(x_test.cuda().view(data_n, -1), recons.view(data_n, -1), p=2).diag())
        print(f"L2 Error: {l2:.5f}")

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

        print(f"Resnet32 clean acc: {accuracy(resnet_clean_preds, y_test):.3f} - "
              f"recons acc: {accuracy(resnet_recon_preds, y_test):.3f}")

        vgg16_clean_preds = torch.argmax(vgg16_bn(x_test), dim=1)
        vgg16_recon_preds = torch.argmax(vgg16_bn(recons), dim=1)

        print(f"VGG16 clean acc: {accuracy(vgg16_clean_preds, y_test):.3f} - "
              f"recons acc: {accuracy(vgg16_recon_preds, y_test):.3f}")


if __name__ == '__main__':

    """
    [ORIGINAL]
    Resnet32 clean acc: 0.930 - recons acc: 0.928
    VGG16 clean acc: 0.934 - recons acc: 0.933
    
    [REPRODUCTION]
    Resnet32 clean acc: 0.930 - recons acc: 0.930
    VGG16 clean acc: 0.934 - recons acc: 0.935
    
    [3SCALES_1GROUP]
    L2 Error: 0.64144
    Resnet32 clean acc: 0.930 - recons acc: 0.922
    VGG16 clean acc: 0.934 - recons acc: 0.929
    
    [3SCALES_1GROUP DETERMINISTIC!]
    L2 Error: 0.41055
    Resnet32 clean acc: 0.930 - recons acc: 0.922
    VGG16 clean acc: 0.934 - recons acc: 0.929
    
    [3SCALES_1GROUP_OURS]
    Resnet32 clean acc: 0.930 - recons acc: 0.906
    VGG16 clean acc: 0.934 - recons acc: 0.922
    """

    IS_OURS = False
    CKPT_NVAE = '/media/dserez/runs/NVAE/cifar10/best/3scales_1group.pt'
    # CKPT_NVAE = '/media/dserez/runs/NVAE/cifar10/ours/cifar10_lr=5e-3_noNF_large/epoch=399.pt'
    ADV_BASE_PATH = '/media/dserez/datasets/cifar10/'
    TORCH_HOME = '/media/dserez/runs/adversarial/CNNs/'
    main()
