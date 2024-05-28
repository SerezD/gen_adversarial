import torch.nn as nn
from torch import Tensor

from torchvision.models import resnet50, vgg16
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.vgg import VGG16_Weights


class ResNet(nn.Module):
    def __init__(self, n_classes: int, get_weights: bool = True) -> None:

        super().__init__()

        weights = ResNet50_Weights.DEFAULT if get_weights else None
        self.model = resnet50(weights=weights)

        # build a 3-layer projector
        prev_dim = self.model.fc.weight.shape[1]
        self.model.fc = nn.Sequential(
                            nn.Linear(prev_dim, prev_dim, bias=False), # first layer
                            nn.BatchNorm1d(prev_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(prev_dim, n_classes))  # output layer

    def forward(self, x: Tensor) -> Tensor:

        return self.model(x)


class Vgg(nn.Module):
    def __init__(self, n_classes: int, get_weights: bool = True) -> None:

        super().__init__()

        weights = VGG16_Weights.DEFAULT if get_weights else None
        self.model = vgg16(weights=weights)

        # build a 3-layer projector
        prev_dim = self.model.classifier[0].weight.shape[1]
        self.model.classifier = nn.Linear(prev_dim, n_classes)  # output layer

    def forward(self, x: Tensor) -> Tensor:

        return self.model(x)