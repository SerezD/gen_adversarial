import torch.nn as nn
from torch import Tensor

from torchvision.models import resnet50


class ResNet(nn.Module):
    def __init__(self,) -> None:

        super().__init__()

        num_classes = 64

        self.model = resnet50(pretrained=True)

        # build a 3-layer projector
        prev_dim = self.model.fc.weight.shape[1]
        self.model.fc = nn.Sequential(
                            nn.Linear(prev_dim, prev_dim, bias=False), # first layer
                            nn.BatchNorm1d(prev_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(prev_dim, num_classes))  # output layer

    def forward(self, x: Tensor) -> Tensor:

        return self.model(x)