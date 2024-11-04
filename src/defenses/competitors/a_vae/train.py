import argparse
import math
import os

from tqdm import tqdm

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from model import StyledGenerator, Discriminator


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(dataset, batch_size, image_size=4):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset.transform = transform
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=8)

    return loader


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups[:2]:
        group['lr'] = lr


def train(args, dataset, generator, discriminator):

    step = int(math.log2(args.img_size)) - 2
    resolution = 4 * 2 ** step

    loader = sample_data(dataset, args.batch_default, resolution)
    data_loader = iter(loader)

    pbar = tqdm(range(3000000))

    requires_grad(generator, False)
    requires_grad(discriminator, True)

    adjust_lr(g_optimizer, 0.004)
    adjust_lr(d_optimizer, 0.004)

    gen_loss_val, kl_loss_val = 0, 0

    for i in pbar:

        # save model
        if i % 8000 == 1:
            torch.save(
                {
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                },
                f'{CKPT_PATH}/train-iter-{i}.pt',
            )

            torch.save(
                g_running.state_dict(), f'{CKPT_PATH}/{str(i + 1).zfill(6)}.pt'
            )

        # get data
        try:
            real_image, label = next(data_loader)
        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image, label = next(data_loader)

        # skip last batch
        if real_image.size(0) != args.batch_default:
            continue

        real_image = real_image.cuda()

        # Discriminator Pass
        discriminator.zero_grad()

        real_predict = discriminator(real_image)
        real_predict = torch.neg_(real_predict.mean() - 0.001 * (real_predict ** 2).mean())
        real_predict.backward()

        _, _, fake_image = generator(F.avg_pool2d(real_image, KERNEL_SIZE))
        fake_predict = discriminator(fake_image)

        fake_predict = fake_predict.mean()
        fake_predict.backward()

        # compute gradient penalty
        eps = torch.rand(args.batch_default, 1, 1, 1).cuda()
        x_hat = eps * real_image.data + (1 - eps) * fake_image.data
        x_hat.requires_grad = True
        hat_predict = discriminator(x_hat)
        grad_x_hat = grad(
            outputs=hat_predict.sum(), inputs=x_hat, create_graph=True
        )[0]
        grad_penalty = (
            (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
        ).mean()
        grad_penalty = 10 * grad_penalty
        grad_penalty.backward()

        d_optimizer.step()

        # save for log
        grad_penalty_val = grad_penalty.item()
        disc_loss_val = (fake_predict.item() + real_predict.item()) / 2

        # Generator Pass every n_critic
        if (i + 1) % args.n_critic == 0:

            generator.zero_grad()

            requires_grad(generator, True)
            requires_grad(discriminator, False)

            m, v, fake_image = generator(F.avg_pool2d(real_image, KERNEL_SIZE))
            predict = discriminator(fake_image)

            rec_loss = -predict.mean()
            kl_loss = -0.5 * torch.mean(-v.exp() - torch.pow(m, 2) + v + 1)
            loss = kl_loss + rec_loss

            loss.backward()
            g_optimizer.step()
            accumulate(g_running, generator)

            kl_loss_val = kl_loss.item()
            gen_loss_val = rec_loss.item()

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        # save some images
        if i % 200 == 0:

            images = [real_image[:8], fake_image[:8]]

            utils.save_image(
                torch.cat(images, 0),
                f'{PLOTS_PATH}/{str(i + 1).zfill(6)}.png',
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
            )

        state_msg = (
            f'GENERATOR: rec {gen_loss_val:.3f}; KL: {kl_loss_val:.5f}; '
            f'DISC: loss {disc_loss_val:.3f}; penalty: {grad_penalty_val:.3f}'  # Consistent: {cs_loss_val:.3f};
        )

        pbar.set_description(state_msg)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--path', type=str, help='path to training set')

    parser.add_argument('--img_size', type=int, help='initial image size, defining the experiment',
                        choices=[64, 128, 256])

    parser.add_argument('--resume', default="", type=str, help='checkpoint')

    args = parser.parse_args()
    args.lr = 1e-3

    # Params depending on image size
    if args.img_size == 64:
        KERNEL_SIZE = 2
        base_path = 'a_vae_celeba_identities'
    elif args.img_size == 128:
        KERNEL_SIZE = 4
        base_path = 'a_vae_cars'
    elif args.img_size == 256:
        KERNEL_SIZE = 8
        base_path = 'a_vae_celeba_gender'
    else:
        raise NotImplementedError

    args.n_critic = 1  # gen steps every

    PLOTS_PATH = f'{base_path}/plots'
    CKPT_PATH = f'{base_path}/checkpoints'

    if not os.path.exists(PLOTS_PATH):
        os.makedirs(PLOTS_PATH)

    if not os.path.exists(CKPT_PATH):
        os.makedirs(CKPT_PATH)

    generator = StyledGenerator(args.img_size).cuda()
    discriminator = Discriminator(args.img_size).cuda()

    g_running = StyledGenerator(args.img_size).cuda()
    g_running.train(False)

    g_optimizer = optim.Adam(
        [{'params': generator.generator.parameters()},
         {'params': generator.encoder.parameters()}
        ],
        lr=args.lr, betas=(0., 0.99)
    )
    g_optimizer.add_param_group(
        {
            'params': generator.style.parameters(),
            'lr': args.lr * 0.01,
            'mult': 0.01,
        }
    )
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0., 0.99))

    if args.resume:
        checkpoint = torch.load(args.resume)
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        print('Load state dict: %s.' % args.checkpoint.split('/')[-1])

    dataset = datasets.ImageFolder(args.path)
    args.batch_default = 32

    train(args, dataset, generator, discriminator)

