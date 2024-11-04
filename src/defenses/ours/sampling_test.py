import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from data.datasets import ImageLabelDataset
from src.defenses.ours.models import CarsTypeClassifier, TransStyleGanDefenseModel


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def main():

    # load pre-trained model
    base_classifier = CarsTypeClassifier(CLASSIFIER_PATH, device)
    defense_model = TransStyleGanDefenseModel(base_classifier, AUTOENCODER_PATH, INTERPOLATION_ALPHAS,
                                              alpha_attenuation=0.8, device=device)

    # dataloader
    dataloader = DataLoader(ImageLabelDataset(folder=IMAGES_PATH, image_size=IMAGE_SIZE),
                            batch_size=BATCH_SIZE, shuffle=False)

    for b_idx, (images, _) in enumerate(tqdm(dataloader)):

        if b_idx == 1:
            break

        images = torch.clip(images.to(device), 0.0, 1.0)

        reconstructions = defense_model(images, preds_only=False)[1]

        display = torch.cat((images, reconstructions), dim=0)
        display = make_grid(display, nrow=BATCH_SIZE).permute(1, 2, 0).cpu().numpy()
        plt.imshow(display)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':

    CLASSIFIER_PATH = '/media/dserez/runs/adversarial/CNNs/resnext50_cars_128_types/best.pt'
    AUTOENCODER_PATH = '/media/dserez/runs/stylegan2/inversions/style_transformer_cars.pt'
    IMAGE_SIZE = 128
    IMAGES_PATH = '/media/dserez/datasets/StanfordCars/subset/validation/'
    BATCH_SIZE = 8

    # INTERPOLATION_ALPHAS = [0.] * 16
    INTERPOLATION_ALPHAS = [0.009607359798384785, 0.03806023374435663, 0.08426519384872738, 0.1464466094067262,
                            0.22221488349019886, 0.3086582838174551, 0.40245483899193585, 0.49999999999999994,
                            0.5975451610080641, 0.6913417161825448, 0.777785116509801, 0.8535533905932737,
                            0.9157348061512727, 0.9619397662556434, 0.9903926402016152, 1.0]



    main()

