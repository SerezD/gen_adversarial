import argparse
import os
import torch
from einops import pack, rearrange
import foolbox as fb
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from data.datasets import ImageLabelDataset
from src.defenses.models import Cifar10ResnetModel, CelebAResnetModel, Cifar10VGGModel, Cifar10NVAEDefenseModel, \
    CelebAStyleGanDefenseModel


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def main():

    # load pre-trained model
    base_classifier = Cifar10VGGModel(CLASSIFIER_PATH, device)
    defense_model = Cifar10NVAEDefenseModel(base_classifier, AUTOENCODER_PATH, device, RESAMPLE_FROM, temperature=0.6)

    # dataloader
    dataloader = DataLoader(ImageLabelDataset(folder=IMAGES_PATH, image_size=IMAGE_SIZE),
                            batch_size=BATCH_SIZE, shuffle=False)

    for b_idx, (images, labels) in enumerate(tqdm(dataloader)):

        images = torch.clip(images.to(device), 0.0, 1.0)
        labels = labels.to(device)

        reconstructions = defense_model.purify(images)

        display = torch.cat((images, reconstructions), dim=0)
        display = make_grid(display, nrow=BATCH_SIZE).permute(1, 2, 0).cpu().numpy()
        plt.imshow(display)
        plt.axis('off')
        plt.show()

        break


if __name__ == '__main__':

    CLASSIFIER_PATH = '/media/dserez/runs/adversarial/CNNs/hub/cifar10_vgg16_bn-6ee7ea24.pt'
    AUTOENCODER_PATH = '/media/dserez/runs/NVAE/cifar10/ours/8x3/epoch=599.pt'
    RESAMPLE_FROM = 16
    IMAGE_SIZE = 32
    IMAGES_PATH = '/media/dserez/datasets/cifar10/validation/'
    BATCH_SIZE = 8

    main()

