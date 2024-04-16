import argparse
from matplotlib import pyplot as plt
import numpy as np
import os
import json


def parse_args():

    parser = argparse.ArgumentParser('Visualize results of "00_class_information" experiments')

    parser.add_argument('--results_path', type=str,
                        required=True, help='.json file with pre-computed accuracies')

    parser.add_argument('--cnn_type', type=str, choices=['Resnet-32', 'Vgg-16', 'Resnet-50'],
                        help='name of CNN that has been evaluated')

    parser.add_argument('--model_type', type=str, choices=['NVAE_3x8', 'StyleGan2'],
                        help='Evaluated generative model')

    args = parser.parse_args()

    # check file exists
    if not os.path.exists(args.results_path):
        raise FileNotFoundError(f'File not Found: {args.results_path}')

    return args


def main(json_file: str, cnn_type: str, model_type: str):

    with open(json_file, 'r') as f:
        accuracies = json.load(f)

    # write table
    text_lines = [
        'latent & top-1 accuracy for alpha (multicol) \\\\ \n',
        'index & 0.0 & 0.1 & 0.2 & 0.3 & 0.4 & 0.5 & 0.6 & 0.7 & 0.8 & 0.9 & 1.0 \\\\ \n'
    ]

    for i, latent in enumerate(accuracies.keys()):

        table_row = f'{latent} & '

        latents_accuracies = []

        for alpha in accuracies[latent].keys():

            latents_accuracies.append(accuracies[latent][alpha])
            table_row += f'{accuracies[latent][alpha]:.3f} & '

        text_lines.append(f'{table_row[:-2]} \\\\ \n')

    with open(f'./00_class_info_{cnn_type}_{model_type}.txt', 'w') as f:
        f.writelines(text_lines)

    # plot histogram of accuracies
    plot_accuracies = []
    for i, latent in enumerate(accuracies.keys()):

        if latent == 'all':
            continue

        plot_accuracies.append(accuracies[latent]['1.00'])

    plt.bar(np.arange(len(plot_accuracies)), plot_accuracies)
    plt.xlabel('Latent Index')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracies when changing latents of {model_type} on {cnn_type}')
    plt.savefig(f'./00_class_info_{cnn_type}_{model_type}.png')
    plt.close()


if __name__ == '__main__':

    arguments = parse_args()

    main(json_file=arguments.results_path,
         cnn_type=arguments.cnn_type,
         model_type=arguments.model_type)
