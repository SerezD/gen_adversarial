import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import ImageLabelDataset
from src.defenses.models import CelebAResnetModel, CelebAStyleGanDefenseModel


class AlphaEvaluator:
    def __init__(self, args, device):

        self.device = device

        # load model
        if args.classifier_type == 'resnet-50':
            args.image_size = 256
            base_classifier = CelebAResnetModel(args.classifier_path, device)
            alphas = [0. for _ in range(18)]
            self.defense_model = CelebAStyleGanDefenseModel(base_classifier, args.autoencoder_path, alphas,
                                                            initial_noise_eps=0.0, apply_gaussian_blur=False,
                                                            device=device)
        else:
            raise ValueError(f'Unknown classifier type: {args.classifier_type}')

        # get dataloader
        dataset = ImageLabelDataset(folder=args.adv_images_path, image_size=args.image_size)
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    @torch.no_grad()
    def objective_function(self, alphas):
        """
        One Epoch of Defense Model with given alphas on the pre-computed adversarial set
        """
        self.defense_model.interpolation_alphas = alphas.cpu().tolist()

        accuracy = None

        for idx, batch in enumerate(tqdm(self.dataloader)):

            x, y = batch[0].to(self.device), batch[1].to(self.device)

            # average result
            n_times_preds = [torch.argmax(self.defense_model(x), dim=-1) for _ in range(6)]
            preds = torch.mode(torch.stack(n_times_preds, dim=0), dim=0).values

            accuracy = torch.eq(preds, y) if accuracy is None else torch.cat((accuracy, preds), dim=0)

            # if idx > 4:
            #     break

        accuracy = torch.mean(accuracy.to(torch.float32)).item()
        return 1 - accuracy