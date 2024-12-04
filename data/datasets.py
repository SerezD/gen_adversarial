import pathlib
from torchvision.transforms import Compose, ToTensor, Resize, RandomHorizontalFlip
from torch.utils.data import Dataset
import torch

from PIL import Image


class ImageNameLabelDataset(Dataset):

    def __init__(self, folder: str, image_size: int):

        self.samples = sorted(list(pathlib.Path(folder).rglob('*.png')) + list(pathlib.Path(folder).rglob('*.jpg')) +
                              list(pathlib.Path(folder).rglob('*.bmp')) + list(pathlib.Path(folder).rglob('*.JPEG')))

        self.samples = [i.absolute().as_posix() for i in self.samples]

        labels_as_str = [i.split('/')[-2] for i in self.samples]
        class_names = sorted(list(set(labels_as_str)))
        self.img_labels = torch.tensor([class_names.index(s) for s in labels_as_str])

        self.transforms = Compose([ToTensor(), Resize((image_size, image_size), antialias=True)])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        x = Image.open(self.samples[idx]).convert('RGB')
        y = self.img_labels[idx]

        return self.transforms(x), self.samples[idx].split('/')[-2:], y


class ImageLabelDataset(Dataset):

    def __init__(self, folder: str, image_size: int):

        self.samples = sorted(list(pathlib.Path(folder).rglob('*.png')) + list(pathlib.Path(folder).rglob('*.jpg')) +
                              list(pathlib.Path(folder).rglob('*.bmp')) + list(pathlib.Path(folder).rglob('*.JPEG')))

        self.samples = [i.absolute().as_posix() for i in self.samples]

        labels_as_str = [i.split('/')[-2] for i in self.samples]
        class_names = sorted(list(set(labels_as_str)))
        self.img_labels = torch.tensor([class_names.index(s) for s in labels_as_str])

        self.transforms = Compose([ToTensor(), Resize((image_size, image_size), antialias=True)])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        x = Image.open(self.samples[idx]).convert('RGB')
        y = self.img_labels[idx]

        return self.transforms(x), y
