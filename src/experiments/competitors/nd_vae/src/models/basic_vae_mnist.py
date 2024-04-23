
import matplotlib.pyplot as plt
import numpy as np
import random


'''Based on the following tutorial: https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71
code at: https://github.com/ttchengab/VAE'''

"""
Import necessary libraries to create a variational autoencoder
The code is mainly developed using the PyTorch library
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image



"""
A Convolutional Variational Autoencoder, made to match that in the Defence-VAE paper: https://github.com/lxuniverse/defense-vae/blob/master/black_box/vae_models.py
"""
class VAE(nn.Module):
    def __init__(self, imgChannels=1, featureDim=4096, zDim=128):
        super(VAE, self).__init__()
        self.encoding_dim = zDim
        # Vae model made to match that in the Defence-VAE code base
        # Encoder
        self.encConv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, bias= False)
        self.encConv1_bn = nn.BatchNorm2d(64)
        self.encConv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=3, bias= False)
        self.encConv2_bn = nn.BatchNorm2d(64)
        self.encConv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias= False)
        self.encConv3_bn = nn.BatchNorm2d(128)
        self.encConv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias= False)
        self.encConv4_bn = nn.BatchNorm2d(256)
        # Latent space
        self.mu_layer = nn.Linear(4096, self.encoding_dim)
        self.logvar_layer = nn.Linear(4096, self.encoding_dim)

        # Decoder
        self.fc3 = nn.Linear(128, 4096)
        self.fc3_bn = nn.BatchNorm1d(4096)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias= False)
        self.deconv1_bn = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias= False)
        self.deconv2_bn = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=3, bias= False)
        self.deconv3_bn = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encoder(self, x):
        out = self.relu(self.encConv1_bn(self.encConv1(x)))
        out = self.relu(self.encConv2_bn(self.encConv2(out)))
        out = self.relu(self.encConv3_bn(self.encConv3(out)))
        out = self.relu(self.encConv4_bn(self.encConv4(out)))
        h1 = out.view(out.size(0), -1)

        # mu and logVar respectively
        mu = self.mu_layer(h1)

        logvar = self.logvar_layer(h1)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into transpose convolutional layers
        # The generated output is the same size of the original input
        h3 = self.relu(self.fc3(z))
        out = h3.view(h3.size(0), 256, 4, 4)
        out = self.relu(self.deconv1_bn(self.deconv1(out)))
        out = self.relu(self.deconv2_bn(self.deconv2(out)))
        out = self.relu(self.deconv3_bn(self.deconv3(out)))
        out = self.sigmoid(self.deconv4(out))
        return out

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        decoded_z = self.decoder(z)

        return decoded_z, mu, logvar
    
    def loss(self, recon_x, x, mu, logvar):
        
        BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD, BCE, KLD

def train_vae_mnist(EPOCHS=10):
    """
    Determine if any GPUs are available
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    """
    Initialize Hyperparameters
    """
    batch_size = 128
    learning_rate = 1e-3
    num_epochs = EPOCHS


    """
    Create dataloaders to feed data into the neural network
    Default MNIST dataset is used and standard train/test split is performed
    """
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                        transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
        batch_size=1)

    """
    Initialize the network and the Adam optimizer
    """
    net = VAE().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)



    """
    Training the network for a given number of epochs
    The loss after every epoch is printed
    """
    for epoch in range(num_epochs + 5):
        for idx, data in enumerate(train_loader, 0):
            imgs, _ = data
            imgs = imgs.to(device)

            # Feeding a batch of images into the network to obtain the output image, mu, and logVar
            out, mu, logVar = net(imgs)

            # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
            #kl_divergence = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
            #loss = F.binary_cross_entropy(out, imgs, size_average=False) + kl_divergence
            loss, bce, kld = net.loss(out,imgs,mu,logVar)

            # Backpropagation based on the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch {}: Loss {}'.format(epoch, loss))



    """
    The following part takes a random image from test loader to feed into the VAE.
    Both the original image and generated image from the distribution are shown.
    """
    sample_count = 10
    image_count = 1
    net.eval()
    with torch.no_grad():
        for data in random.sample(list(test_loader), sample_count):
            imgs, _ = data
            imgs = imgs.to(device)
            img = np.transpose(imgs[0].cpu().numpy(), [1,2,0])
            plt.subplot(121)
            plt.imshow(np.squeeze(img))
            out, mu, logVAR = net(imgs)
            outimg = np.transpose(out[0].cpu().numpy(), [1,2,0])
            plt.subplot(122)
            plt.imshow(np.squeeze(outimg))
            plt.savefig('../../results/vae_examples/vae_examples'+str(image_count)+'.png')
            plt.clf()
            image_count += 1


    # Save model parameters
    torch.save(net.state_dict(), '../../model_parameters/basic_vae_model_mnist.pth')




###############################################################################
###################     Fashion MNIST Version     #############################
###############################################################################

def train_vae_fashion_mnist(EPOCHS=10):
    """
    Determine if any GPUs are available
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    """
    Initialize Hyperparameters
    """
    batch_size = 128
    learning_rate = 1e-3
    num_epochs = EPOCHS


    """
    Create dataloaders to feed data into the neural network
    Default MNIST dataset is used and standard train/test split is performed
    """
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=True, download=True,
                        transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=False, transform=transforms.ToTensor()),
        batch_size=1)

    """
    Initialize the network and the Adam optimizer
    """
    net = VAE().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)



    """
    Training the network for a given number of epochs
    The loss after every epoch is printed
    """
    for epoch in range(num_epochs):
        for idx, data in enumerate(train_loader, 0):
            imgs, _ = data
            imgs = imgs.to(device)

            # Feeding a batch of images into the network to obtain the output image, mu, and logVar
            out, mu, logVar = net(imgs)

            # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
            kl_divergence = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
            loss = F.binary_cross_entropy(out, imgs, size_average=False) + kl_divergence

            # Backpropagation based on the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch {}: Loss {}'.format(epoch, loss))



    """
    The following part takes a random image from test loader to feed into the VAE.
    Both the original image and generated image from the distribution are shown.
    """
    sample_count = 10
    image_count = 1
    net.eval()
    with torch.no_grad():
        for data in random.sample(list(test_loader), sample_count):
            imgs, _ = data
            imgs = imgs.to(device)
            img = np.transpose(imgs[0].cpu().numpy(), [1,2,0])
            plt.subplot(121)
            plt.imshow(np.squeeze(img))
            out, mu, logVAR = net(imgs)
            outimg = np.transpose(out[0].cpu().numpy(), [1,2,0])
            plt.subplot(122)
            plt.imshow(np.squeeze(outimg))
            plt.savefig('../../results/vae_examples/vae_examples_fashion'+str(image_count)+'.png')
            plt.clf()
            image_count += 1


    # Save model parameters
    torch.save(net.state_dict(), '../../model_parameters/basic_vae_model_fashion_mnist.pth')





#Un comment to run training through this script
#train_vae_mnist()
#train_vae_fashion_mnist()