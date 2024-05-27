import argparse
import math
import os

from tqdm import tqdm

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad
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
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

def sample_data(dataset, batch_size, image_size=4):
    transform = transforms.Compose(
        [
            # transforms.CenterCrop(180), # TODO REMOVED
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
        #mult = group.get('mult', 1)
        group['lr'] = lr
        # print(group['lr'])


def train(args, dataset, generator, discriminator):
    step = int(math.log2(args.init_size)) - 2
    resolution = 4 * 2 ** step
    loader = sample_data(
        dataset, args.batch.get(resolution, args.batch_default), resolution
    )
    data_loader = iter(loader)

    mseloss = torch.nn.MSELoss().cuda()
    # mseloss = torch.nn.L1Loss().cuda()
    pbar = tqdm(range(150000))  # TODO this was originally 3000000

    requires_grad(generator, False)
    requires_grad(discriminator, True)

    disc_loss_val = 0
    gen_loss_val = 0
    rec_loss_val = 0
    grad_loss_val = 0
    adjust_lr(g_optimizer, 0.004)
    adjust_lr(d_optimizer, 0.004)
    used_sample = 0

    for i in pbar:
        discriminator.zero_grad()

        alpha = min(1., 20. / args.phase * (used_sample + 1))
        # alpha = 0 if used_sample < 1000000 else 1

        if i % 8000 == 1:
            torch.save(
                {
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    # 'e_optimizer': e_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                },
                f'{CKPT_PATH}/train-iter-{i}.pt',
            )

            torch.save(
                g_running.state_dict(), f'{CKPT_PATH}/{str(i + 1).zfill(6)}.pt'
            )

            # adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
            # adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

        try:
            real_image, label = next(data_loader)

        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image, label = next(data_loader)

        used_sample += real_image.shape[0]

        b_size = real_image.size(0)
        real_image = real_image.cuda()

        real_predict = discriminator(real_image)
        real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
        (-real_predict).backward()

        _, _, fake_image = generator(F.avg_pool2d(real_image, KERNEL_SIZE))
        fake_predict = discriminator(fake_image)


        fake_predict = fake_predict.mean()
        fake_predict.backward()

        eps = torch.rand(b_size, 1, 1, 1).cuda()
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
        grad_loss_val = grad_penalty.item()
        disc_loss_val = fake_predict.item()
        real_loss_val = -real_predict.item()


        d_optimizer.step()

        if (i + 1) % n_critic == 0:
            generator.zero_grad()

            requires_grad(generator, True)
            requires_grad(discriminator, False)

            m, v, fake_image = generator(F.avg_pool2d(real_image, KERNEL_SIZE))
            predict = discriminator(fake_image)

            if args.loss == 'wgan-gp':
                d_loss = -predict.mean()

            elif args.loss == 'r1':
                d_loss = F.softplus(-predict).mean()

            cs_loss = mseloss(fake_image, real_image)
            kl_loss = -0.5 * torch.mean(-v.exp() - torch.pow(m, 2) + v + 1)
            kl_loss_val = kl_loss.item()
            loss = kl_loss + d_loss
            gen_loss_val = d_loss.item()
            rec_loss_val = cs_loss.item()

            loss.backward()
            g_optimizer.step()
            accumulate(g_running, generator)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        if i % 200 == 0:
            images = []

            gen_i, gen_j = args.gen_sample.get(resolution, (8, 4))
            with torch.no_grad():
                for j in range(2):
                    k = j % 4
                    images.append(
                        real_image[k*gen_j:k*gen_j+4].cpu()
                    )
                for j in range(gen_j):
                    k = j % 2
                    images.append(
                        generator(
                            F.avg_pool2d(real_image[k*gen_j:k*gen_j+4], KERNEL_SIZE)
                        )[2].data.cpu()
                    )


            utils.save_image(
                torch.cat(images, 0),
                f'{PLOTS_PATH}/{str(i + 1).zfill(6)}.png',
                nrow=gen_i,
                normalize=True,
                value_range=(-1, 1),
            )

        # if alpha == 0:
        #     state_msg = (
        #         f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; KL: {kl_loss_val:.6f}; '
        #         f'Alpha: {alpha:.3f};'  # Consistent: {cs_loss_val:.3f};
        #     )
        # else:
        state_msg = (
            f'Size: {4 * 2 ** step}; Gen: 1.ganloss {gen_loss_val:.3f}; recloss {rec_loss_val:.3f}; KLloss: {kl_loss_val:.5f}; '
            f'D: 1.fake {disc_loss_val:.3f}; 2.real {real_loss_val:.3f}; Grad: {grad_loss_val:.3f}'  # Consistent: {cs_loss_val:.3f};
        )
        # state_msg = (
        #     f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; '
        #     f'KL: {kl_loss_val:.5f}; Alpha: {alpha:.3f}; Grad: {grad_loss_val:.3f}'  # Consistent: {cs_loss_val:.3f};
        # )
        # else:
        #     state_msg = (
        #         f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; '
        #         f'Alpha: {alpha:.3f};'  # Consistent: {cs_loss_val:.3f};
        #     )

        pbar.set_description(state_msg)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    parser.add_argument('--path', type=str,
                        default='../dataset/CASIA-WebFace', help='path of specified dataset')
    parser.add_argument(
        '--n_gpu', type=int, default=1, help='number of gpu used for training'
    )
    parser.add_argument(
        '--phase',
        type=int,
        default=1000000,
        help='number of samples used for each training phases',
    )
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', default=True, type=bool, help='use lr scheduling')
    parser.add_argument('--init_size', default=128, type=int, help='initial image size')
    parser.add_argument('--max_size', default=128, type=int, help='max image size')
    # /mnt/sda/zjl/workspace/style-based-gan/checkpoint_v9/train-iter-8.model
    parser.add_argument('--resume',
                        default="",
                        type=str,
                        help='checkpoint')
    parser.add_argument(
        '--mixing', action='store_true', help='use mixing regularization'
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='wgan-gp',
        choices=['wgan-gp', 'r1'],
        help='class of gan loss',
    )
    parser.add_argument(
        '-d',
        '--data',
        default='folder',
        type=str,
        choices=['folder', 'lsun'],
        help=('Specify dataset. Currently Image Folder and LSUN is supported'),
    )

    args = parser.parse_args()

    # TODO
    #   KERNEL_SIZE for DOWNSAMPLING
    #   CELEB-A 256 // 8 -> 32
    #   CIFAR-10  32 // 2 -> 16
    KERNEL_SIZE = 1 if args.init_size == 32 else 8
    code_size = 512
    n_critic = 1

    base_path = f'a_vae_cifar10' if args.init_size == 32 else f'a_vae_celeba'

    PLOTS_PATH = f'{base_path}/plots'
    CKPT_PATH = f'{base_path}/checkpoints'

    if not os.path.exists(PLOTS_PATH):
        os.makedirs(PLOTS_PATH)

    if not os.path.exists(CKPT_PATH):
        os.makedirs(CKPT_PATH)

    generator = StyledGenerator(code_size, args.init_size // KERNEL_SIZE, args.init_size).cuda()
    discriminator = Discriminator(args.init_size).cuda()
    g_running = StyledGenerator(code_size, args.init_size // KERNEL_SIZE, args.init_size).cuda()
    g_running.train(False)

    class_loss = nn.CrossEntropyLoss()

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
        # e_optimizer.load_state_dict(checkpoint['e_optimizer'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        print('Load state dict: %s.' % args.checkpoint.split('/')[-1])

    # accumulate(g_running, generator, 0)

    if args.data == 'folder':
        dataset = datasets.ImageFolder(args.path)

    elif args.data == 'lsun':
        dataset = datasets.LSUNClass(args.path, target_transform=lambda x: 0)

    if args.sched:
        args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        args.batch = {4: 64, 8: 32, 16: 32, 32: 32, 64: 32, 128: 16}

    else:
        args.lr = {}
        args.batch = {}

    args.gen_sample = {512: (8, 4), 1024: (4, 2)}

    args.batch_default = 32 if args.init_size == 256 else 256

    train(args, dataset, generator, discriminator)
