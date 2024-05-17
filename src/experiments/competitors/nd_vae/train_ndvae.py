import pathlib

import numpy as np
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize

from src.experiments.competitors.nd_vae.src.models.NVAE import Defence_NVAE
from src.experiments.competitors.nd_vae.src.NVAE_defense_training import NVAE_Defense_training
from src.experiments.competitors.nd_vae.src.data_utils import ImgDataset

def main():

    # define NVAE model
    model = Defence_NVAE(NVAE_PARAMS['x_channels'],
                         NVAE_PARAMS['encoding_channels'],
                         NVAE_PARAMS['pre_proc_groups'],
                         NVAE_PARAMS['scales'],
                         NVAE_PARAMS['groups'],
                         NVAE_PARAMS['cells'],
                         IMAGE_SIZE[0])

    # define Dataset
    train_dataset = ImgDataset(TRAIN_FOLDER, ADV_FOLDER, IMAGE_SIZE, noisy_input=USE_NOISE, noise_max=NOISE_MAX)

    NVAE_Defense_training(TEST_NAME, model, train_dataset, BATCH_SIZE, EPOCHS, LR)


if __name__ == '__main__':

    # ADAPTED from CONFIG.PY

    # # CELEBA-HQ
    # TEST_NAME = 'celeba'
    # IMAGE_SIZE = (256, 256)
    # TRAIN_FOLDER = '/media/dserez/datasets/celeba_hq_gender/train'
    # ADV_FOLDER = '/media/dserez/datasets/celeba_hq_gender/fsgm/train'
    # EPOCHS = 50
    # LR = 1e-3
    # TUNING_EPOCHS = 50
    # DEF_VAE_EPOCHS = 50
    # BATCH_SIZE = 32
    # LOAD_DATA = False
    # CLASSIFIER_PRETRAINED = False
    # CLASSIFIER_PRETUNED = False
    # VAE_PRETRAINED = False
    # REDUCED_DATA = True  # limits data to 300 elements for testing
    #
    # NVAE_PARAMS = {
    #     'x_channels': 3,
    #     'pre_proc_groups': 2,
    #     'encoding_channels': 16,
    #     'scales': 2,
    #     'groups': 4,
    #     'cells': 2
    # }
    #
    # NOISE_MAX = .1
    # USE_NOISE = True

    # CIFAR-10 Resnet-32
    TEST_NAME = 'cifar10_resnet32'
    IMAGE_SIZE = (32, 32)
    TRAIN_FOLDER = '/media/dserez/datasets/cifar10/train'
    ADV_FOLDER = '/media/dserez/datasets/cifar10/fsgm_resnet32/train'
    EPOCHS = 50
    LR = 1e-3
    TUNING_EPOCHS = 50
    DEF_VAE_EPOCHS = 50
    BATCH_SIZE = 128
    LOAD_DATA = False
    CLASSIFIER_PRETRAINED = False
    CLASSIFIER_PRETUNED = False
    VAE_PRETRAINED = False
    REDUCED_DATA = True  # limits data to 300 elements for testing

    NVAE_PARAMS = {
        'x_channels': 3,
        'pre_proc_groups': 2,
        'encoding_channels': 16,
        'scales': 1,
        'groups': 4,
        'cells': 2
    }

    NOISE_MAX = .05
    USE_NOISE = True

    main()