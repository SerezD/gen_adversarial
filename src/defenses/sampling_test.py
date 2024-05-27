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
from src.defenses.models import CelebAResnetModel, NVAEDefenseModel


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def main():

    # load pre-trained model
    base_classifier = CelebAResnetModel(CLASSIFIER_PATH, device)
    defense_model = NVAEDefenseModel(base_classifier, AUTOENCODER_PATH, INTERPOLATION_ALPHAS,
                                     device=device, temperature=TEMPERATURE)

    # dataloader
    dataloader = DataLoader(ImageLabelDataset(folder=IMAGES_PATH, image_size=IMAGE_SIZE),
                            batch_size=BATCH_SIZE, shuffle=False)

    for b_idx, (images, _) in enumerate(tqdm(dataloader)):

        if b_idx == 1:
            break

        images = torch.clip(images.to(device), 0.0, 1.0)

        reconstructions = defense_model.purify(images)

        display = torch.cat((images, reconstructions), dim=0)
        display = make_grid(display, nrow=BATCH_SIZE).permute(1, 2, 0).cpu().numpy()
        plt.imshow(display)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':

    CLASSIFIER_PATH = '/media/dserez/runs/adversarial/CNNs/resnet50_celeba_hq_gender/best.pt'
    AUTOENCODER_PATH = '/media/dserez/runs/NVAE/celeba64/last.pt'
    IMAGE_SIZE = 64
    IMAGES_PATH = '/media/dserez/datasets/celeba_hq_gender/validation/'
    BATCH_SIZE = 8

    # INTERPOLATION_ALPHAS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #                         0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
    #                         0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8)

    # INTERPOLATION_ALPHAS = (0.04, 0.08, 0.12, 0.17, 0.21, 0.25, 0.29, 0.33,
    #                         0.38, 0.42, 0.46, 0.50, 0.54, 0.58, 0.62, 0.67,
    #                         0.71, 0.75, 0.79, 0.83, 0.88, 0.92, 0.96, 1.0)

    INTERPOLATION_ALPHAS = (0.00, 0.02, 0.04, 0.07, 0.10, 0.15, 0.20, 0.25,
                            0.31, 0.37, 0.43, 0.50, 0.57, 0.63, 0.69, 0.75,
                            0.80, 0.85, 0.90, 0.93, 0.96, 0.98, 1.00, 1.00)

    TEMPERATURE = 0.8

    main()

