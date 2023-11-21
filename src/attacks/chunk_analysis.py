import math
import os
from collections import OrderedDict

import torch
from einops import pack
from robustbench import load_cifar10

from src.NVAE.original.model import AutoEncoder
from src.NVAE.original.utils import get_arch_cells
from src.attacks.latent_walkers import NonlinearWalker


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
    nvae = nvae.cuda().eval()

    # get size of each chunk
    # this is latent_size * latent_resolution ** 2
    # repeat it based on num groups per scale (latent resolution)
    chunk_size = [nvae.num_latent_per_group * (32 // (2 ** (i + 1))) ** 2
                  for i in range(nvae.num_latent_scales)]
    chunk_size.reverse()
    chunk_size = [s for s in chunk_size for _ in range(nvae.num_groups_per_scale)]

    # Walker
    nn_walker = NonlinearWalker(n_chunks=sum(nvae.groups_per_scale), chunk_size=chunk_size,
                                init_weights=False)

    checkpoint = torch.load(CKPT_WALKER, map_location='cpu')
    state_dict = OrderedDict()

    # remove module (no ddp)
    for key in checkpoint.keys():
        state_dict['.'.join(key.split('.')[1:])] = checkpoint[key]

    nn_walker.load_state_dict(state_dict)
    nn_walker = nn_walker.to('cuda').eval()

    # load cifar10 test set
    x_test, _ = load_cifar10(data_dir=ADV_BASE_PATH)
    x_test = x_test.to('cuda')
    data_n = x_test.shape[0]

    # compute chunks and differences
    test_set_noise = torch.empty((0, sum(nvae.groups_per_scale), 1), device='cuda')

    for b in range(0, data_n, BATCH_SIZE):

        with torch.no_grad():

            x = x_test[b:b+BATCH_SIZE]

            # encode
            chunks = nvae.encode(x)

            # get adversarial chunks
            adversarial_chunks = []
            l2_dist_vector = torch.empty((BATCH_SIZE, 0, 1), device='cuda')

            for i, c in enumerate(chunks):
                adv_c = nn_walker(c, i)
                r = int(math.sqrt(adv_c.shape[1] // nvae.num_latent_per_group))
                adversarial_chunks.append(adv_c.view(BATCH_SIZE, nvae.num_latent_per_group, r, r))

                # compute L2 distance
                l2_dist = torch.cdist(c, adv_c, p=2).diag().view(-1, 1, 1)
                l2_dist_vector, _ = pack([l2_dist_vector, l2_dist], 'b * d')

            test_set_noise, _ = pack([test_set_noise, l2_dist_vector], '* n d')


            # decode to get adversarial images (need grad for backprop)
            # nvae_logits = nvae.decode(adversarial_chunks)
            # adversarial_samples = nvae.decoder_output(nvae_logits).sample()

    # compute mean and std per chunk!
    means = torch.mean(test_set_noise, dim=0)
    stds = torch.std(test_set_noise, dim=0)

    # print TABLE
    print('chunk N & mean & std \\\\')
    print('\\toprule')
    for i in range(means.shape[0]):
        print(f'{i} & {means[i].item():.2f} & {stds[i].item():.2f} \\\\')
        if (i + 1) < means.shape[0] and (i + 1) % 2 == 0:
            print('\\midrule')
    print('\\bottomrule')

if __name__ == '__main__':

    CKPT_NVAE = '/media/dserez/runs/NVAE/cifar10/best/ours_weighted.pt'
    CKPT_WALKER = '/media/dserez/runs/adversarial/cifar10/whitebox/ep_39.pt'
    ADV_BASE_PATH = '/media/dserez/code/adversarial/'
    TORCH_HOME = '/media/dserez/runs/classification/cifar10/'

    BATCH_SIZE = 100

    main()