"""
. Load precomputed reconstructions
. For one batch:
    plot them
    compute diff image
. For all:
    histogram where bin is mean diff on all test set per each pixel
    histogram where bin is max diff on all test set per each pixel
"""

import os
import numpy as np
import torch
from torchvision.utils import make_grid
from einops import rearrange, pack

from matplotlib import pyplot as plt
from robustbench import load_cifar10


def main():

    if os.path.exists(IMAGES_FILE):
        all_recons = torch.tensor(np.load(IMAGES_FILE))

        x_test, _ = load_cifar10(data_dir=ADV_BASE_PATH)
        all_recons, _ = pack([x_test.unsqueeze(1), all_recons], 'b * c h w')
        all_recons = all_recons.numpy()

        # REVERSE CHUNKS COLUMNS!
        chunks = np.flip(all_recons[:, 2:], axis=1)
        all_recons[:, 2:] = chunks

    else:
        raise AttributeError(f'IMAGES_FILE: {IMAGES_FILE}\ndoes not exist!')

    imgs_to_plot = all_recons[:BATCH_SIZE]
    diff_to_plot = np.abs(np.repeat(np.expand_dims(all_recons[:BATCH_SIZE, 0], axis=1),
                                    all_recons.shape[1], axis=1) - all_recons[:BATCH_SIZE])

    # normalize each image between 0__1
    for b in range(diff_to_plot.shape[0]):
        for n in range(diff_to_plot.shape[1]):
            max_values = np.expand_dims(np.max(np.reshape(diff_to_plot[b, n], (3, -1)), axis=1), axis=(1, 2))
            if max_values[0] == 0.:
                continue
            diff_to_plot[b, n] /= max_values

    n = all_recons.shape[1]
    imgs_to_plot = make_grid(rearrange(torch.tensor(imgs_to_plot), 'b n c h w -> (b n) c h w'), nrow=n, padding=5)
    diff_to_plot = make_grid(rearrange(torch.tensor(diff_to_plot), 'b n c h w -> (b n) c h w'), nrow=n, padding=5)

    fig, [l, c, r] = plt.subplots(1, 3, figsize=(12, 8))
    l.imshow(rearrange(imgs_to_plot, 'c h w -> h w c').numpy())
    l.axis(False)
    l.set_title('Pixel Space Images')

    c.imshow(rearrange(diff_to_plot, 'c h w -> h w c').numpy())
    c.axis(False)
    c.set_title('difference image over 3 dims')

    # one scale diff
    im = r.imshow(rearrange(torch.mean(diff_to_plot, dim=0, keepdim=True), 'c h w -> h w c').numpy())
    r.axis(False)
    r.set_title('difference mean over 3 dims')
    plt.colorbar(im)

    plt.savefig(f'{SAVE_PATH}difference_images.png')
    plt.close(fig)

    diff_images = np.abs(np.repeat(np.expand_dims(all_recons[:, 0], axis=1), n, axis=1) - all_recons)

    for i in range(1, n):

        # compute mean and max both on images and channels, to get 1024 bins
        mean_diff_per_pixel = np.mean(np.mean(diff_images[:, i], axis=0), axis=0).reshape(-1)
        max_diff_per_pixel = np.max(np.max(diff_images[:, i], axis=0), axis=0).reshape(-1)

        fig, (l, r) = plt.subplots(1, 2, figsize=(24, 12))
        l.bar(np.arange(1024), mean_diff_per_pixel, width=3.0, edgecolor='black', linewidth=0.1)
        l.set_title('mean pixel differences')
        r.bar(np.arange(1024), max_diff_per_pixel, width=3.0, edgecolor='black', linewidth=0.1)
        r.set_title('max pixel differences')
        name = "reconstructions" if i == 1 else f"chunk_{i - 2}"
        plt.suptitle(f'Histograms for "{name}"')
        plt.savefig(f'{SAVE_PATH}histo_{name}.png')
        plt.close(fig)


if __name__ == '__main__':

    ADV_BASE_PATH = '../media/'
    IMAGES_FILE = f'{ADV_BASE_PATH}reconstructions_ours_weighted.npy'
    SAVE_PATH = f'{ADV_BASE_PATH}/images/ours_weighted/'
    BATCH_SIZE = 20

    main()
