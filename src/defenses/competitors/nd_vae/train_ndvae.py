import argparse

from src.defenses.competitors.nd_vae.modules.models.NVAE import Defence_NVAE
from src.defenses.competitors.nd_vae.modules.NVAE_defense_training import NVAE_Defense_training
from src.defenses.competitors.nd_vae.modules.data_utils import ImgDataset


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

    parser = argparse.ArgumentParser('TRAIN ND VAE model')

    parser.add_argument('--images_path', type=str, required=True,
                        help='Base path to dataset')

    parser.add_argument('--type', type=str, choices=['celeba256', 'celeba64', 'cars128'],
                        help='what to train')

    args = parser.parse_args()

    TRAIN_FOLDER = f'{args.images_path}/train/'
    ADV_FOLDER = f'{args.images_path}/ndvae_adversaries/'
    TEST_NAME = args.type

    if args.type == 'celeba256':

        IMAGE_SIZE = (256, 256)
        EPOCHS = 50
        LR = 1e-3
        BATCH_SIZE = 32

        NVAE_PARAMS = {
            'x_channels': 3,
            'pre_proc_groups': 2,
            'encoding_channels': 16,
            'scales': 2,
            'groups': 4,
            'cells': 2
        }

        NOISE_MAX = .1
        USE_NOISE = True

    elif args.type == 'celeba64':

        IMAGE_SIZE = (64, 64)
        EPOCHS = 400
        LR = 1e-4
        BATCH_SIZE = 256

        NVAE_PARAMS = {
            'x_channels': 3,
            'pre_proc_groups': 2,
            'encoding_channels': 8,
            'scales': 1,
            'groups': 2,
            'cells': 4
        }

        NOISE_MAX = .05
        USE_NOISE = True

    elif args.type == 'cars128':

        IMAGE_SIZE = (128, 128)
        EPOCHS = 100
        LR = 1e-3
        BATCH_SIZE = 32

        NVAE_PARAMS = {
            'x_channels': 3,
            'pre_proc_groups': 2,
            'encoding_channels': 16,
            'scales': 2,
            'groups': 2,
            'cells': 4
        }

        NOISE_MAX = .1
        USE_NOISE = True

    main()
