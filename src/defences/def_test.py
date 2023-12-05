"""
take different autoattack attacks.
encode adversaries with nvae and check distance from origin w.r.t clean
"""
import math

import torch
import os
import time

from einops import pack
from kornia.enhance import Normalize
from matplotlib import pyplot as plt
from torchmetrics import Accuracy
from tqdm import tqdm

from autoattack.autopgd_base import APGDAttack
from autoattack.fab_pt import FABAttack_PT
from autoattack.square import SquareAttack
from robustbench import load_cifar10
from torchvision.utils import make_grid

from src.NVAE.original.model import AutoEncoder
from src.NVAE.original.utils import get_arch_cells

@torch.no_grad()
def main():

    # load nvae pretrained cifar10
    checkpoint = torch.load(CKPT_NVAE, map_location='cpu')

    # get and update args
    args = checkpoint['args']
    args.num_mixture_dec = 10
    args.batch_size = BATCH_SIZE

    # init model and load
    arch_instance = get_arch_cells(args.arch_instance)
    nvae = AutoEncoder(args, None, arch_instance)
    nvae.load_state_dict(checkpoint['state_dict'], strict=False)
    nvae = nvae.to('cuda:0').eval()

    # load cifar10
    x_test, y_test = load_cifar10(data_dir=ADV_BASE_PATH)
    data_n = x_test.shape[0]
    # data_n = 100
    x_test = x_test[:data_n]
    y_test = y_test[:data_n]

    # 1. sample recons vs mean recons L2 error
    # guess -> recons with mean gives more accurate stuff ?
    # x_rec_sample = torch.empty((0, 3, 32, 32), device='cuda:0')
    # x_rec_mean = torch.empty((0, 3, 32, 32), device='cuda:0')
    #
    # for b in tqdm(range(0, data_n, BATCH_SIZE)):
    #
    #     x = x_test[b:b + BATCH_SIZE]
    #
    #     # encode and get chunks, mu
    #     sample_chunks, mean_chunks = nvae.encode(x.to('cuda:0'))
    #     x_rec_sample, _ = pack([x_rec_sample, nvae.decoder_output(nvae.decode(sample_chunks)).sample()], '* c h w')
    #     x_rec_mean, _ = pack([x_rec_mean, nvae.decoder_output(nvae.decode(mean_chunks)).sample()], '* c h w')
    #
    # # Calculate L2 Distance
    # sample_l2 = (x_test.to('cuda:0').view(data_n, -1) - x_rec_sample.view(data_n, -1)).pow(2).sum(dim=1).sqrt().mean()
    # mean_l2 = (x_test.to('cuda:0').view(data_n, -1) - x_rec_mean.view(data_n, -1)).pow(2).sum(dim=1).sqrt().mean()
    #
    # print(f'SAMPLE L2 ON CIFAR10 TEST SET: {sample_l2}.\n'
    #       f'MEAN L2 ON CIFAR10 TEST SET: {mean_l2}.')

    # 2. TEST CHUNK SHIFT after attack
    # cnn to test
    os.environ["TORCH_HOME"] = TORCH_HOME
    resnet32 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True)
    resnet32 = resnet32.to('cuda:1').eval()

    cnn_preprocess = Normalize(mean=torch.tensor([0.507, 0.4865, 0.4409], device='cuda:1'),
                               std=torch.tensor([0.2673, 0.2564, 0.2761], device='cuda:1'))
    predict_pass = lambda x: resnet32(cnn_preprocess(x))
    accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1).to('cuda:0')

    metric, bound = 'L2', 0.5
    attack = APGDAttack(predict_pass, norm=metric, eps=bound)


if __name__ == '__main__':

    CKPT_NVAE = '/media/dserez/runs/NVAE/cifar10/best/ours_weighted.pt'
    ADV_BASE_PATH = '/media/dserez/code/adversarial/'
    TORCH_HOME = '/media/dserez/runs/classification/cifar10/'
    OUT_FILE = f'{ADV_BASE_PATH}images/'

    BATCH_SIZE = 100

    # CKPT_NVAE = '/home/dserez/gen_adversarial/loading/ours_weighted.pt'
    # ADV_BASE_PATH = '/home/dserez/gen_adversarial/loading/'
    # TORCH_HOME = '/home/dserez/gen_adversarial/loading/'
    # OUT_FILE = f'{ADV_BASE_PATH}defense_images/'
    #
    # BATCH_SIZE = 200

    main()