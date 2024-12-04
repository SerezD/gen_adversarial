import torch.nn as nn
from torch import Tensor

from torchvision.models import resnet50, vgg11_bn, resnext50_32x4d
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.vgg import VGG11_BN_Weights
from torchvision.models.resnet import ResNeXt50_32X4D_Weights


class ResNet(nn.Module):
    def __init__(self, n_classes: int, get_weights: bool = True) -> None:

        super().__init__()

        weights = ResNet50_Weights.DEFAULT if get_weights else None
        self.model = resnet50(weights=weights)

        # build a 3-layer projector
        prev_dim = self.model.fc.weight.shape[1]
        self.model.fc = nn.Sequential(
                            nn.Linear(prev_dim, prev_dim, bias=False),
                            nn.BatchNorm1d(prev_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(prev_dim, n_classes))

    def forward(self, x: Tensor) -> Tensor:

        return self.model(x)


class Vgg(nn.Module):
    def __init__(self, n_classes: int, get_weights: bool = True) -> None:

        super().__init__()

        weights = VGG11_BN_Weights.DEFAULT if get_weights else None
        self.model = vgg11_bn(weights=weights)

        # build a 3-layer projector
        prev_dim = self.model.classifier[0].weight.shape[1]
        self.model.classifier = nn.Sequential(
                            nn.Linear(prev_dim, prev_dim, bias=False),
                            nn.BatchNorm1d(prev_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(prev_dim, n_classes))

    def forward(self, x: Tensor) -> Tensor:

        return self.model(x)


class ResNext(nn.Module):
    def __init__(self, n_classes: int, get_weights: bool = True) -> None:

        super().__init__()

        weights = ResNeXt50_32X4D_Weights.DEFAULT if get_weights else None
        self.model = resnext50_32x4d(weights=weights)

        # build a 3-layer projector
        prev_dim = self.model.fc.weight.shape[1]
        self.model.fc = nn.Sequential(
                            nn.Linear(prev_dim, prev_dim, bias=False),
                            nn.BatchNorm1d(prev_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(prev_dim, n_classes))

    def forward(self, x: Tensor) -> Tensor:

        return self.model(x)
