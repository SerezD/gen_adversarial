import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from data.datasets import CoupledDataset
from src.NVAE.original.model import AutoEncoder
from src.NVAE.original.utils import get_arch_cells
import matplotlib.pyplot as plt


def interpolate(data_dir: str, nvae_checkpoint: str):

    # data
    dataloader = DataLoader(CoupledDataset(folder=data_dir, image_size=32), batch_size=1, shuffle=False)

    # NVAE
    # load nvae pretrained cifar10
    checkpoint = torch.load(nvae_checkpoint, map_location='cpu')

    # get and update args
    args = checkpoint['args']
    args.num_mixture_dec = 10

    # init model and load
    arch_instance = get_arch_cells(args.arch_instance)
    nvae = AutoEncoder(args, None, arch_instance)
    nvae.load_state_dict(checkpoint['state_dict'], strict=False)
    nvae.cuda().eval()

    with torch.no_grad():

        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        for (src_x, src_y, trg_x, trg_y)in dataloader:

            src_x = src_x.cuda()
            trg_x = trg_x.cuda()

            src_chunks = nvae.encode(src_x)[2]
            trg_chunks = nvae.encode(trg_x)[2]

            samples_pixel = torch.empty((0, 3, 32, 32), device=src_x.device)
            samples_latent = torch.empty((0, 3, 32, 32), device=src_x.device)

            for alpha in alphas:
                # sample = [(1 - alpha) * s + alpha * a for (s, a) in zip(src_chunks, adv_chunks)]
                # sample = nvae.decode(sample)

                interpolation = (1 - alpha) * src_x + alpha * trg_x
                samples_pixel = torch.cat([samples_pixel, interpolation], dim=0)

                interpolation = [(1 - alpha) * s + alpha * a for (s, a) in zip(src_chunks, trg_chunks)]
                logits = nvae.decode(interpolation)
                interpolation = nvae.decoder_output(logits).mean()
                samples_latent = torch.cat([samples_latent, interpolation], dim=0)

            samples_pixel = make_grid(samples_pixel, nrow=11).permute(1, 2, 0).cpu().numpy()
            samples_latent = make_grid(samples_latent, nrow=11).permute(1, 2, 0).cpu().numpy()

            fig, ax = plt.subplots(2, 1, figsize=(16, 12))
            ax[0].imshow(samples_pixel)
            ax[0].axis(False)
            ax[1].imshow(samples_latent)
            ax[1].axis(False)
            plt.show()


if __name__ == '__main__':

    interpolate(data_dir="/media/dserez/datasets/cifar10/validation/",
                nvae_checkpoint="/media/dserez/runs/NVAE/cifar10/best/3scales_1group.pt")
