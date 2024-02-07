from argparse import Namespace

import torch
from einops import pack
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from kornia.enhance import normalize

import os

from torchvision.utils import make_grid
from tqdm import tqdm

from data.datasets import CoupledDataset
from src.StyleGan.models.hyperstyle import HyperStyle
from src.classifier.model import ResNet


def get_model_conf(filepath: str):
    import yaml

    # load params
    with open(filepath, 'r', encoding='utf-8') as stream:
        params = yaml.safe_load(stream)

    return params


def main():

    # Load resnet
    ckpt = torch.load(RESNET_PATH, map_location='cpu')
    resnet = ResNet()
    resnet.load_state_dict(ckpt['state_dict'])
    resnet.cuda(0).eval()

    # Load StyleNetWork
    ckpt = torch.load(DECODER_PATH, map_location='cpu')

    opts = ckpt['opts']
    opts['checkpoint_path'] = DECODER_PATH
    opts['load_w_encoder'] = True
    opts['w_encoder_checkpoint_path'] = ENCODER_PATH
    opts = Namespace(**opts)

    autoencoder = HyperStyle(opts)
    autoencoder.cuda(0).eval()

    # load dataset
    dataloader = DataLoader(CoupledDataset(folder=DATA_PATH, image_size=256), batch_size=8, shuffle=False)

    # structure = {'chunk_n' : {alpha_: {'recons': tensor, 'preds': tensor}, ...}, ...}
    save_dict = {}
    alpha = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], device='cuda:0')
    chunks_to_test = ('all', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    n_valid = 0

    with torch.no_grad():

        for batch_idx, batch in enumerate(tqdm(dataloader)):

            # if batch_idx == 2:
            #     break

            # first images and labels, to_interpolate
            x1, y1, x2, y2 = [i.cuda(0) for i in batch]

            # reconstruct test set
            b, c, h, w = x1.shape

            x = normalize(torch.cat([x1, x2]),
                          torch.tensor([0.5, 0.5, 0.5], device='cuda:0'),
                          torch.tensor([0.5, 0.5, 0.5], device='cuda:0'))

            # get reconstructions, adjusted_weights, codes
            inputs = x.clone()
            recons, latent, weights_deltas, codes = None, None, None, None

            for _ in range(4):

                recons, latent, weights_deltas, codes, _ = autoencoder.forward(inputs,
                                                                               randomize_noise=False,
                                                                               return_latents=True,
                                                                               return_weight_deltas_and_codes=True,
                                                                               weights_deltas=weights_deltas,
                                                                               y_hat=recons, codes=codes
                                                                               )

            # denormalize and split
            recons = (recons + 1) / 2
            x1_recons, x2_recons = torch.split(recons, b)

            # measure accuracy and keep only valid samples.
            x1_preds = torch.argmax(resnet(x1_recons), dim=1)
            x2_preds = torch.argmax(resnet(x2_recons), dim=1)

            valid_samples = torch.logical_and(
                torch.eq(x1_preds, y1),
                torch.eq(x2_preds, y2)
            )

            # keep deltas of src and split chunks
            weights_deltas = [w[:b] if w is not None else w for w in weights_deltas]
            chunks_x1, chunks_x2 = torch.split(latent, b)

            # keep only valid samples
            weights_deltas = [w[valid_samples] if w is not None else w for w in weights_deltas]
            chunks_x1 = chunks_x1[valid_samples]
            chunks_x2 = chunks_x2[valid_samples]
            y1 = y1[valid_samples]
            n_valid += y1.shape[0]

            # interpolate at different alpha terms and check when class is changing
            interpolated_codes = chunks_x1.unsqueeze(1).repeat(1, len(alpha), 1, 1)
            interpolation = chunks_x2.unsqueeze(1).repeat(1, len(alpha), 1, 1)
            alpha = alpha.view(1, -1, 1, 1)
            interpolated_codes = (1 - alpha) * interpolated_codes + alpha * interpolation

            for n in chunks_to_test:

                if batch_idx == 0:
                    # structure = {'chunk_n' : {alpha_: {'recons': tensor, 'preds': tensor}, ...}, ...}
                    save_dict[f'chunk_{n}'] = {}

                for i in range(alpha.shape[1]):

                    # get recons
                    if n == 'all':
                        recons = autoencoder.decode(interpolated_codes[:, i], weights_deltas, resize=True)
                        recons = (recons + 1) / 2
                    else:
                        initial_codes = interpolated_codes[:, 0].clone()  # alpha 0 == no_interpolation
                        initial_codes[:, n] = interpolated_codes[:, i, n].clone()
                        recons = autoencoder.decode(initial_codes, weights_deltas, resize=True)
                        recons = (recons + 1) / 2

                    # get preds
                    preds = torch.argmax(resnet(recons), dim=1)

                    a = alpha[0, i].item()

                    # append to res
                    if batch_idx == 0:

                        # structure = {'chunk_n' : {alpha_: {'recons': tensor, 'preds': tensor, y}, ...}, ...}
                        save_dict[f'chunk_{n}'][f'alpha_{a:.2f}'] = {
                            'recons': recons[0].unsqueeze(0).cpu(),
                            'preds': preds.cpu(),
                            'targets': y1.cpu()
                        }
                    else:
                        save_dict[f'chunk_{n}'][f'alpha_{a:.2f}']['preds'] = torch.cat(
                            [save_dict[f'chunk_{n}'][f'alpha_{a:.2f}']['preds'], preds.cpu()]
                        )
                        save_dict[f'chunk_{n}'][f'alpha_{a:.2f}']['targets'] = torch.cat(
                            [save_dict[f'chunk_{n}'][f'alpha_{a:.2f}']['targets'], y1.cpu()]
                        )

        # print out all results
        final_picture = torch.empty((0, alpha.shape[1], 3, 256, 256))

        for k in save_dict.keys():

            all_alphas = save_dict[k]
            picture_row = torch.empty((0, 3, 256, 256))

            print('#' * 20)
            print(f'chunks: {k}')
            for a in all_alphas.keys():
                picture_row = torch.cat((picture_row, all_alphas[a]['recons']), dim=0)
                res = (all_alphas[a]['preds'] == all_alphas[a]['targets']).sum() / n_valid
                print(f'alpha: {a} - accuracy: {res}')

            final_picture = torch.cat((final_picture, picture_row.unsqueeze(0)))

        display = make_grid(final_picture.view(-1, 3, 256, 256), nrow=alpha.shape[1]).permute(1,2,0).cpu().numpy()
        fig, ax = plt.subplots(figsize=(16, 24))
        ax.imshow(display, interpolation='nearest')
        ax.axis(False)
        plt.tight_layout()
        plt.savefig('./class_change_StyleGAN.png')
        plt.close()



if __name__ == '__main__':

    DATA_PATH = '/media/dserez/datasets/afhq/wild/validation/'
    RESNET_PATH = '/media/dserez/runs/adversarial/CNNs/resnet50_afhq_wild/best.pt'
    ENCODER_PATH = f'/media/dserez/runs/stylegan2/inversions/encoder_afhq_wild.pt'
    DECODER_PATH = f'/media/dserez/runs/stylegan2/inversions/hyperstyle_inverter_afhq_wild.pt'

    main()
