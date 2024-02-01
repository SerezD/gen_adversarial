import pathlib
from torchvision.transforms import Compose, ToTensor, Resize, RandomHorizontalFlip
from torch.utils.data import Dataset
import torch

from PIL import Image


class ImageDataset(Dataset):

    def __init__(self, folder: str, image_size: int, ffcv: bool = False):

        self.samples = sorted(list(pathlib.Path(folder).rglob('*.png')) + list(pathlib.Path(folder).rglob('*.jpg')) +
                              list(pathlib.Path(folder).rglob('*.bmp')) + list(pathlib.Path(folder).rglob('*.JPEG')))
        self.ffcv = ffcv

        if 'train' in folder:
            self.transforms = Compose([ToTensor(), RandomHorizontalFlip(),
                                       Resize((image_size, image_size), antialias=True)])
        else:
            self.transforms = Compose([ToTensor(), Resize((image_size, image_size), antialias=True)])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        # path to string
        image_path = self.samples[idx].absolute().as_posix()

        image = Image.open(image_path).convert('RGB')

        return (image, ) if self.ffcv else self.transforms(image)


class ImageLabelDataset(Dataset):

    def __init__(self, folder: str, image_size: int, ffcv: bool = False):

        self.samples = sorted(list(pathlib.Path(folder).rglob('*.png')) + list(pathlib.Path(folder).rglob('*.jpg')) +
                              list(pathlib.Path(folder).rglob('*.bmp')) + list(pathlib.Path(folder).rglob('*.JPEG')))

        self.samples = [i.absolute().as_posix() for i in self.samples]
        self.img_labels = torch.tensor([int(i.split('/')[-2]) for i in self.samples])

        self.ffcv = ffcv

        if 'train' in folder:
            self.transforms = Compose([ToTensor(), RandomHorizontalFlip(),
                                       Resize((image_size, image_size), antialias=True)])
        else:
            self.transforms = Compose([ToTensor(), Resize((image_size, image_size), antialias=True)])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        x = Image.open(self.samples[idx]).convert('RGB')
        y = self.img_labels[idx]

        return (x, y) if self.ffcv else (self.transforms(x), y)


class CoupledDataset(Dataset):
    def __init__(self, folder: str, image_size: int, seed: int = 1):

        # all image paths
        self.samples = sorted(list(pathlib.Path(folder).rglob('*.png')) + list(pathlib.Path(folder).rglob('*.jpg')) +
                              list(pathlib.Path(folder).rglob('*.bmp')) + list(pathlib.Path(folder).rglob('*.JPEG')))
        self.samples = [i.absolute().as_posix() for i in self.samples]
        self.img_labels = torch.tensor([int(i.split('/')[-2]) for i in self.samples])

        self.transforms = Compose([ToTensor(), Resize((image_size, image_size), antialias=True)])
        self.seed = seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        src_x = self.transforms(Image.open(self.samples[idx]).convert('RGB'))
        src_y = self.img_labels[idx]

        cand_trg = torch.nonzero(torch.not_equal(self.img_labels, torch.ones_like(self.img_labels) * src_y)).squeeze(1)
        trg_idx = cand_trg[torch.randint(len(cand_trg), (1,), generator=torch.Generator().manual_seed(self.seed)).item()]

        trg_x = self.transforms(Image.open(self.samples[trg_idx]).convert('RGB'))
        trg_y = self.img_labels[trg_idx]

        return src_x, src_y, trg_x, trg_y


# if __name__ == '__main__':
#     d = CoupledDataset(folder='/media/dserez/datasets/cifar10', image_size=32)
#     d.__getitem__(30)
