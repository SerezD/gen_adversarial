'''Drawn from the following tutorial https://nextjournal.com/gkoehler/pytorch-mnist'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt


#TODO CONVERT TO MODEL(S) FROM PAPER
'''
class Guide_Net(nn.Module):
    def __init__(self, channels_in):
        super(Guide_Net, self).__init__()
        self.input_layer = nn.Linear(channels_in,32)
        self.bn1 = nn.BatchNorm1d(32)
        self.input2 = nn.Linear(32,64)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv1 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1,padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=2,padding=0)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv2_drop = nn.Dropout2d(p=.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.input_layer(x)))
        x = F.relu(self.bn2(self.input2(x)))
        x = F.relu(self.bn3(self.conv1(x)))
        x = F.relu(self.conv2_drop(self.bn4(self.conv2(x))))
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=.5, training=self.training)
        x = self.fc2(x)
        return x #F.softmax(x)
'''

class Guide_Net(nn.Module):
    def __init__(self, channels_in):
        super(Guide_Net, self).__init__()
        self.input_layer = nn.Linear(channels_in,8)
        self.bn1 = nn.BatchNorm1d(8)
        self.dense1 = nn.Linear(8,16)
        self.bn2 = nn.BatchNorm1d(16)
        self.dense2 = nn.Linear(16,32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dense3 = nn.Linear(32,16)
        self.bn4 = nn.BatchNorm1d(16)
        self.out = nn.Linear(16,2)


    def forward(self, x):
        x = F.relu(self.bn1(self.input_layer(x)))
        x = F.relu(self.bn2(self.dense1(x)))
        x = F.relu(self.bn3(self.dense2(x)))
        x = F.dropout(x, p=.25, training=self.training)
        x = F.relu(self.bn4(self.dense3(x)))
        x = self.out(x)
        return x #F.softmax(x)


