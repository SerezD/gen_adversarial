import torch
from einops import pack
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from kornia.enhance import normalize

import os

from torchvision.utils import make_grid
from tqdm import tqdm

from data.datasets import CoupledDataset
from src.NVAE.mine.distributions import DiscMixLogistic
from src.NVAE.mine.model import AutoEncoder
from src.NVAE.original.utils import get_arch_cells


def get_model_conf(filepath: str):
    import yaml

    # load params
    with open(filepath, 'r', encoding='utf-8') as stream:
        params = yaml.safe_load(stream)

    return params


def main():

    # # create model and move it to GPU with id rank
    # # load nvae pretrained cifar10
    # checkpoint = torch.load(CKPT_NVAE, map_location='cpu')
    #
    # # get and update args
    # args = checkpoint['args']
    # args.num_mixture_dec = 10
    # args.batch_size = 100
    #
    # # init model and load
    # arch_instance = get_arch_cells(args.arch_instance)
    # nvae = AutoEncoder(args, arch_instance)
    # nvae.load_state_dict(checkpoint['state_dict'], strict=False)
    # nvae = nvae.cuda().eval()

    checkpoint = torch.load(CKPT_NVAE, map_location='cpu')

    config = checkpoint['configuration']

    # create model and move it to GPU with id rank
    nvae = AutoEncoder(config['autoencoder'], config['resolution'])

    nvae.load_state_dict(checkpoint['state_dict'])
    nvae.cuda().eval()


    # load dataset
    dataloader = DataLoader(CoupledDataset(folder=DATA_PATH, image_size=32), batch_size=128, shuffle=False)

    # load classifiers pretrained cifar10
    os.environ["TORCH_HOME"] = TORCH_HOME

    if cnn_type == 'resnet':
        cnn = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True)
        cnn = cnn.cuda().eval()
    else:
        cnn = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
        cnn = cnn.cuda().eval()

    alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    res = torch.empty((0, len(alpha), 3),device='cuda')
    example_images = torch.empty((0, 3, 32, 32), device='cuda')  # alpha_n, 3, 32, 32

    with torch.no_grad():

        for batch_idx, batch in enumerate(tqdm(dataloader)):

            # first images and labels, to_interpolate
            x1, y1, x2, y2 = [i.cuda() for i in batch]

            # TODO
            # # reconstruct cifar10 test set
            # logits = nvae.decode(nvae.encode_deterministic(x1))
            # x1_recons = nvae.decoder_output(logits).mean()
            #
            # logits = nvae.decode(nvae.encode_deterministic(x2))
            # x2_recons = nvae.decoder_output(logits).mean()

            logits = nvae.decode(nvae.encode(x1, deterministic=True))
            x1_recons = DiscMixLogistic(logits).mean()

            logits = nvae.decode(nvae.encode(x2, deterministic=True))
            x2_recons = DiscMixLogistic(logits).mean()

            # measure accuracy and keep only valid samples.
            x1_recons = normalize(x1_recons,
                               mean=torch.tensor([0.507, 0.4865, 0.4409], device='cuda:0'),
                               std=torch.tensor([0.2673, 0.2564, 0.2761], device='cuda:0'))

            x2_recons = normalize(x2_recons,
                               mean=torch.tensor([0.507, 0.4865, 0.4409], device='cuda:0'),
                               std=torch.tensor([0.2673, 0.2564, 0.2761], device='cuda:0'))

            x1_preds = torch.argmax(cnn(x1_recons), dim=1)
            x2_preds = torch.argmax(cnn(x2_recons), dim=1)

            valid_samples = torch.logical_and(
                torch.eq(x1_preds, y1),
                torch.eq(x2_preds, y2)
            )

            x1 = x1[valid_samples]
            y1 = y1[valid_samples]
            x2 = x2[valid_samples]
            y2 = y2[valid_samples]
            n_valid = y1.shape[0]

            # interpolate at different alpha terms and check when class is changing
            # TODO
            # chunks_x1 = nvae.encode_deterministic(x1)
            # chunks_x2 = nvae.encode_deterministic(x2)

            chunks_x1 = nvae.encode(x1, deterministic=True)
            chunks_x2 = nvae.encode(x2, deterministic=True)

            alpha_res = torch.empty((0, 3), device='cuda')

            for i, a in enumerate(alpha):

                # TODO CHANGE THIS
                if chunk_to_test == 0:
                    int_chunks = [(1 - a) * chunks_x1[0] + a * chunks_x2[0]] + chunks_x1[1:]
                elif chunk_to_test == 1:
                    int_chunks = [chunks_x1[0]] + [(1 - a) * chunks_x1[1] + a * chunks_x2[1]] + [chunks_x1[-1]]
                    # WITH 2 chunks
                    # int_chunks = [chunks_x1[0]] + [(1 - a) * chunks_x1[1] + a * chunks_x2[1]]
                elif chunk_to_test == 2:
                    int_chunks = chunks_x1[:-1] + [(1 - a) * chunks_x1[-1] + a * chunks_x2[-1]]
                else:
                    # interpolate all
                    int_chunks = [(1 - a) * c_x1 + a * c_x2 for c_x1, c_x2 in zip(chunks_x1, chunks_x2)]

                logits = nvae.decode(int_chunks)
                # recons = nvae.decoder_output(logits).mean() TODO
                recons = DiscMixLogistic(logits).mean()

                if batch_idx == 0:
                    example_images = torch.cat((example_images, recons[0].unsqueeze(0)), dim=0)

                recons = normalize(recons,
                                   mean=torch.tensor([0.507, 0.4865, 0.4409], device='cuda:0'),
                                   std=torch.tensor([0.2673, 0.2564, 0.2761], device='cuda:0'))
                preds = torch.argmax(cnn(recons), dim=1)
                this_res = torch.stack(
                    [
                        (preds == y1).sum() / n_valid,
                        (preds == y2).sum() / n_valid,
                        (torch.logical_and(preds != y1, preds != y2)).sum() / n_valid
                ], dim=0).view(1, 3)
                alpha_res, _ = pack([alpha_res, this_res], '* d')

            res, _ = pack([res, alpha_res.unsqueeze(0)], '* a d')

            if batch_idx == 0:
                example_images = torch.cat((example_images, x2[0].unsqueeze(0)), dim=0)

        print(f"CHUNK {chunk_to_test}")
        final_res = torch.mean(res, dim=0).cpu().numpy()
        for a, r in zip(alpha, final_res):
            print(f"alpha = {a}, src_class, adv_class, None = {r}")

        numpy_img = make_grid(example_images, nrow=10).permute(1, 2, 0).cpu().numpy()
        plt.imshow(numpy_img)
        plt.axis(False)
        plt.show()


if __name__ == '__main__':

    DATA_PATH = '/media/dserez/datasets/cifar10/validation/'
    TORCH_HOME = '/media/dserez/runs/adversarial/CNNs/'
    cnn_type = 'vgg'  # 'vgg' 'resnet'

    # CKPT_NVAE = f'/media/dserez/runs/NVAE/cifar10/best/3scales_1group_latest.pt'
    CKPT_NVAE = f'/media/dserez/runs/NVAE/cifar10/ours/replica/large_latents.pt'

    # VGG with medium 1 !
    # ALL   0.981 0.943 0.839 0.585 0.232 0.072 0.014 0.000 0.000
    # 0     0.993 0.986 0.975 0.960 0.938 0.911 0.880 0.846 0.806
    # 1     0.990 0.979 0.966 0.944 0.915 0.877 0.832 0.775 0.717
    # 2     0.987 0.973 0.949 0.914 0.864 0.794 0.708 0.625 0.555

    for chunk_to_test in (0, 1, 2, 'all'):
        print('#' * 20)
        print(f'chunk = {chunk_to_test}')
        main()
