from argparse import Namespace
import torch

from src.classifier.model import ResNet, Vgg, ResNext
from src.mlvgms_autoencoders.NVAE.model import AutoEncoder
from src.mlvgms_autoencoders.StyleGan_E4E.psp import pSp
from src.mlvgms_autoencoders.StyleGan_Trans_n.models.style_transformer import StyleTransformer


def load_ResNet50(path: str, device: str, n_classes: int = 2) -> ResNet:

    ckpt = torch.load(path, map_location='cpu')
    resnet = ResNet(n_classes=n_classes, get_weights=False)
    resnet.load_state_dict(ckpt['state_dict'])
    resnet.to(device).eval()
    return resnet


def load_Vgg11(path: str, device: str, n_classes: int = 100) -> Vgg:

    ckpt = torch.load(path, map_location='cpu')
    vgg11 = Vgg(n_classes=n_classes, get_weights=False)
    vgg11.load_state_dict(ckpt['state_dict'])
    vgg11.to(device).eval()
    return vgg11


def load_ResNext50(path: str, device: str, n_classes: int = 4) -> ResNext:

    ckpt = torch.load(path, map_location='cpu')
    resnext = ResNext(n_classes=n_classes, get_weights=False)
    resnext.load_state_dict(ckpt['state_dict'])
    resnext.to(device).eval()
    return resnext


def load_E4EStyleGan(checkpoint_path: str, device: str) -> pSp:

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = ckpt['opts']

    opts['checkpoint_path'] = checkpoint_path
    opts['device'] = device
    opts = Namespace(**opts)

    net = pSp(opts)
    net = net.to(device).eval()
    return net


def load_NVAE(checkpoint_path: str, device: str, temperature: float) -> AutoEncoder:
    """
    :param checkpoint_path: path to NVAE checkpoint containing 'state_dict' and 'configuration'
    :param device: model will be returned in eval mode on this device
    :param temperature: temperature parameter for loading specific state dict
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    config = checkpoint['configuration']

    # create model and move it to GPU with id rank
    nvae = AutoEncoder(config['autoencoder'], config['resolution'])

    nvae.load_state_dict(checkpoint[f'state_dict_temp={temperature}'])
    nvae.to(device).eval()
    return nvae


def load_TranStyleGan(checkpoint_path: str, device: str) -> StyleTransformer:

    # update test options with options used during training
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts = Namespace(**opts)
    opts.checkpoint_path = checkpoint_path

    net = StyleTransformer(opts)
    net.load_weights()
    net.eval().to(device)

    return net
