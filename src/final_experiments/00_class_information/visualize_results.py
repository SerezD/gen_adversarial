import argparse
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle


def parse_args():

    parser = argparse.ArgumentParser('Test top-1 accuracy of CNNs when interpolating images on NVAE chunks')

    # TODO for public release, replace "default=..." with "required=True"

    parser.add_argument('--results_path', type=str,
                        default='../results/class_change_NVAE_large_latents_RESNET32/class_change_accuracies.pickle',
                        help='.pickle file with pre-computed accuracies')

    parser.add_argument('--cnn_type', type=str, choices=['Resnet-32', 'Vgg-16', 'Resnet-50'],
                        help='name of CNN that has been evaluated')

    parser.add_argument('--model_type', type=str, choices=['NVAE', 'StyleGan2'],
                        help='Evaluated generative model')

    args = parser.parse_args()

    # check file exists
    if not os.path.exists(args.results_path):
        raise FileNotFoundError(f'File not Found: {args.results_path}')

    return args


def main(pickle_file: str, cnn_type: str, model_type: str):

    with open(pickle_file, 'rb') as f:
        accuracies = pickle.load(f)

    alphas_x_labels = []
    colors = ['lightcoral', 'darkorange', 'limegreen', 'dodgerblue', 'blueviolet', 'fuchsia',
              'tomato', 'gold', 'teal', 'navy', 'mediumpurple', 'deeppink',
              'brown', 'yellow', 'forestgreen', 'deepskyblue', 'indigo', 'crimson']

    linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (5, (10, 3)),
                  (0, (1, 10)), (0, (5, 10)), (0, (3, 10, 1, 10)),
                  (0, (1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1))]
    tickness = np.arange(1.5, 4.0, 0.5).repeat((len(linestyles)))
    linestyles = linestyles * 4

    print('latent & top-1 accuracy for alpha (multicol) \\\\')
    print('index & 0.0 & 0.1 & 0.2 & 0.3 & 0.4 & 0.5 & 0.6 & 0.7 & 0.8 & 0.9 & 1.0 \\\\')

    for i, latent in enumerate(accuracies.keys()):

        table_row = f'{latent} & '

        latents_accuracies = []

        for alpha in accuracies[latent].keys():

            if i == 0:
                alphas_x_labels.append(float(alpha))

            latents_accuracies.append(accuracies[latent][alpha].item())
            table_row += f'{accuracies[latent][alpha].item():.3f} & '

        print(table_row[:-2] + '\\\\')

        plt.plot(alphas_x_labels, latents_accuracies, linestyle=linestyles[i], linewidth=tickness[i],
        label=f'latent: {latent}', color=colors[i])

    plt.legend()
    plt.xlabel('alpha interpolation')
    plt.ylabel('Top-1 Accuracy')
    plt.title(f'Top-1 Accuracy of {cnn_type} when interpolating \nimages in {model_type} latent spaces')
    plt.show()


if __name__ == '__main__':

    arguments = parse_args()

    main(pickle_file=arguments.results_path,
         cnn_type=arguments.cnn_type,
         model_type=arguments.model_type)
