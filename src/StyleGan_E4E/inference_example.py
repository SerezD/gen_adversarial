import argparse
import torch
from kornia.enhance import normalize, denormalize
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

from data.datasets import ImageDataset
from src.StyleGan_E4E.psp import pSp


def setup_model(checkpoint_path, device='cuda'):

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = ckpt['opts']

    opts['checkpoint_path'] = checkpoint_path
    opts['device'] = device
    opts = argparse.Namespace(**opts)

    net = pSp(opts)
    net.eval()
    net = net.to(device)
    return net, opts


@torch.no_grad()
def main(args):

    net, opts = setup_model(args.ckpt, device)

    generator = net.decoder
    generator.eval()

    images_path = args.images_dir
    print(f"images path: {images_path}")

    test_dataset = ImageDataset(folder=images_path, image_size=256, ffcv=None)
    data_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, drop_last=False)
    mean_vec = torch.tensor([0.5, 0.5, 0.5], device=device)
    std_vec = torch.tensor([0.5, 0.5, 0.5], device=device)

    print(f'dataset length: {len(test_dataset)}')

    for images in data_loader:

        images = normalize(images.to(device), mean=mean_vec, std=std_vec)

        # recons = net(images)
        codes = net.encode(images)
        recons = net.decode(codes)

        display = make_grid(
            denormalize(torch.cat((images, recons), dim=0), mean=mean_vec, std=std_vec),
            nrow=4).permute(1, 2, 0).cpu().numpy()
        plt.imshow(display)
        plt.axis(False)
        plt.tight_layout()
        plt.show()

        break


if __name__ == "__main__":

    device = "cuda"

    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="The directory of the images to be inverted")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="The directory to save the latent codes and inversion images. (default: images_dir")
    parser.add_argument("--batch", type=int, default=1, help="batch size for the generator")
    parser.add_argument("--n_sample", type=int, default=None, help="number of the samples to infer.")
    parser.add_argument("--latents_only", action="store_true", help="infer only the latent codes of the directory")
    parser.add_argument("--align", action="store_true", help="align face images before inference")

    args = parser.parse_args()
    args.images_dir = '/media/dserez/datasets/celeba_hq_gender/test/female/'
    args.save_dir = './results/'
    args.ckpt = '/media/dserez/runs/stylegan2/inversions/e4e_ffhq_encoder.pt'

    main(args)
