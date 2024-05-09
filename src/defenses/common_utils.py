import os
from argparse import Namespace

import torch

from src.hl_autoencoders.NVAE.model import AutoEncoder
from src.hl_autoencoders.StyleGan_E4E.psp import pSp
from src.classifier.model import ResNet


def load_NVAE(checkpoint_path: str, device: str):
    """
    :param checkpoint_path: path to NVAE checkpoint containing 'state_dict' and 'configuration'
    :param device: model will be returned in eval mode on this device
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    config = checkpoint['configuration']

    # create model and move it to GPU with id rank
    nvae = AutoEncoder(config['autoencoder'], config['resolution'])

    nvae.load_state_dict(checkpoint['state_dict'])
    nvae.to(device).eval()
    return nvae


def load_hub_CNN(model_path: str, cnn_type: str, device: str):
    """
    returns a CNN model (resnet32 or vgg16) downloaded from torch hub and pre-trained on CIFAR10.
    model source = https://github.com/chenyaofo/pytorch-cifar-models

    :param model_path: path to model downloaded from hub (if not found, will download it)
    :param cnn_type: choice is 'resnet32' or 'vgg16'
    :param device: model will be returned in eval mode on this device
    """

    os.environ["TORCH_HOME"] = model_path

    if cnn_type == 'resnet32' or cnn_type == 'resnet-32':
        cnn = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True)
    elif cnn_type == 'vgg16' or cnn_type == 'vgg-16':
        cnn = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
    else:
        raise ValueError(f"parameter type = {cnn_type} not recognized.")

    cnn = cnn.to(device).eval()
    return cnn


def load_StyleGan(checkpoint_path: str, device: str):

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = ckpt['opts']

    opts['checkpoint_path'] = checkpoint_path
    opts['device'] = device
    opts = Namespace(**opts)

    net = pSp(opts)
    net = net.to(device).eval()
    return net


def load_ResNet_CelebA(path: str, device: str):

    ckpt = torch.load(path, map_location='cpu')
    resnet = ResNet(get_weights=False)
    resnet.load_state_dict(ckpt['state_dict'])
    resnet.to(device).eval()
    return resnet
