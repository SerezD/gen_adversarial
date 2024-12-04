import argparse
import math
import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import ImageLabelDataset
from src.defenses.ours.models import CelebaGenderClassifier, CelebaIdentityClassifier, CarsTypeClassifier, \
    E4EStyleGanDefenseModel, NVAEDefenseModel, TransStyleGanDefenseModel
from src.defenses.wrappers import EoTWrapper


def get_linear_alphas(n: int) -> list:

    return [i / n for i in range(1, n + 1)]


def get_cosine_alphas(n: int) -> list:

    return [0.5 * (1 - math.cos(math.pi * (i / n))) for i in range(1, n + 1)]


def get_best_combination(folder: str) -> np.array:

    alphas = np.load(f'{folder}/alphas.npy')
    accuracies = np.load(f'{folder}/accuracies.npy')[:, 0]
    return alphas[accuracies.argmax()]


class AlphaEvaluator:
    def __init__(self, args: argparse.Namespace, device: int):

        self.device = device
        self.eot_steps = 32

        # load model
        if args.classifier_type == 'resnet-50':

            args.image_size = 256
            self.alpha_attenuation = 1.

            base_classifier = CelebaGenderClassifier(args.classifier_path, device)

            alphas = [0. for _ in range(18)]
            self.defense_model = E4EStyleGanDefenseModel(base_classifier, args.autoencoder_path, alphas,
                                                         self.alpha_attenuation, device=device).eval()

        elif args.classifier_type == 'vgg-11':

            args.image_size = 64
            self.alpha_attenuation = 0.7

            base_classifier = CelebaIdentityClassifier(args.classifier_path, device)

            alphas = [0. for _ in range(24)]
            self.defense_model = NVAEDefenseModel(base_classifier, args.autoencoder_path, alphas,
                                                  alpha_attenuation=0.7, device=device).eval()

        elif args.classifier_type == 'resnext-50':

            args.image_size = 128
            self.alpha_attenuation = 0.7

            base_classifier = CarsTypeClassifier(args.classifier_path, device)

            alphas = [0. for _ in range(16)]
            self.defense_model = TransStyleGanDefenseModel(base_classifier, args.autoencoder_path, alphas,
                                                           alpha_attenuation=0.7, device=device).eval()

        else:
            raise ValueError(f'Unknown classifier type: {args.classifier_type}')

        self.defense_model = EoTWrapper(self.defense_model, self.eot_steps).eval()

        # get dataloader
        dataset = ImageLabelDataset(folder=args.adv_images_path, image_size=args.image_size)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    @torch.no_grad()
    def objective_function(self, alphas: tocrch.Tensor) -> float:
        """
        One Epoch of Defense Model with given alphas on the pre-computed adversarial set
        """

        # update model with new set of parameters.
        self.defense_model.model.interpolation_alphas = [a * self.alpha_attenuation for a in alphas.cpu().tolist()]

        # compute accuracy
        accuracy = torch.empty((0,), dtype=torch.bool, device=self.device)

        for idx, (x, y) in enumerate(tqdm(self.dataloader)):

            x, y = x.to(self.device), y.to(self.device)

            # eot average result
            preds = self.defense_model(x).argmax(dim=1)

            accuracy = torch.cat((accuracy, torch.eq(preds, y)), dim=0)

        accuracy = torch.mean(accuracy.to(torch.float32)).item()
        return accuracy
