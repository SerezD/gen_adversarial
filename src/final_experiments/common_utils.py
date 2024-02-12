import os
from argparse import Namespace

import torch

from src.NVAE.mine.model import AutoEncoder
from src.StyleGan.models.hyperstyle import HyperStyle
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


def load_hub_CNN(model_path: str, type: str, device: str):
    """
    returns a CNN model (resnet32 or vgg16) downloaded from torch hub and pre-trained on CIFAR10.
    model source = https://github.com/chenyaofo/pytorch-cifar-models

    :param model_path: path to model downloaded from hub (if not found, will download it)
    :param type: choice is 'resnet32' or 'vgg16'
    :param device: model will be returned in eval mode on this device
    """

    os.environ["TORCH_HOME"] = model_path

    if type == 'resnet32' or type == 'resnet-32':
        cnn = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True)
    elif type == 'vgg16' or type == 'vgg-16':
        cnn = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
    else:
        raise ValueError(f"parameter type = {type} not recognized.")

    cnn = cnn.to(device).eval()
    return cnn


def load_StyleGan(encoder_path: str, decoder_path: str, device: str):

    ckpt = torch.load(decoder_path, map_location='cpu')

    opts = ckpt['opts']
    opts['checkpoint_path'] = decoder_path
    opts['load_w_encoder'] = True
    opts['w_encoder_checkpoint_path'] = encoder_path
    opts = Namespace(**opts)

    autoencoder = HyperStyle(opts)
    autoencoder.to(device).eval()
    return autoencoder


def load_ResNet_AFHQ_Wild(path: str, device: str):

    ckpt = torch.load(path, map_location='cpu')
    resnet = ResNet()
    resnet.load_state_dict(ckpt['state_dict'])
    resnet.to(device).eval()
    return resnet
