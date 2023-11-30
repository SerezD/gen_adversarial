import argparse
from kornia.enhance import Normalize
import math
from matplotlib import pyplot as plt
import os

import torch
import numpy as np
from torch import optim
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.backends import cudnn
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as ddp
from torch.utils.data import DistributedSampler, DataLoader

from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
from torchmetrics.functional import accuracy
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

from tqdm import tqdm

from src.attacks.latent_walkers import NonlinearWalker
from src.NVAE.original.model import AutoEncoder
from src.NVAE.original.utils import get_arch_cells

import warnings


def init_run():
    parser = argparse.ArgumentParser('train a non linear latent walker for a white box attack setting.')

    parser.add_argument('--nvae_path', type=str, help='pretrained NVAE path',
                        default='/media/dserez/runs/NVAE/cifar10/best/ours_weighted.pt')
    parser.add_argument('--torch_home', type=str,
                        default='/media/dserez/runs/classification/cifar10/')
    parser.add_argument('--data_dir', type=str,
                        default='/media/dserez/code/adversarial/')

    parser.add_argument('--chunks_to_perturb', type=int, default=None, nargs='+',
                        help='indices of chunks to perturb, starting from 0')

    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--img_size', type=int, default=32)

    parser.add_argument('--save_folder', type=str, default='./walker_runs')
    parser.add_argument('--run_name', type=str, default='whitebox_walker')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4, help='lr for non linear walker')
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    opt = parser.parse_args()

    # create checkpoint and plots folder
    opt.ckpt_folder = os.path.join(opt.save_folder, 'ckpts', opt.run_name)
    opt.plots_folder = os.path.join(opt.save_folder, 'plots', opt.run_name)

    if not os.path.isdir(opt.ckpt_folder):
        os.makedirs(opt.ckpt_folder)

    if not os.path.isdir(opt.plots_folder):
        os.makedirs(opt.plots_folder)

    return opt


def setup(rank, world_size):
    if 'MASTER_ADDR' not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        if rank == 0:
            warnings.warn("Set Environ Variable 'MASTER_ADDR'='localhost'")

    if 'MASTER_PORT' not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
        if rank == 0:
            warnings.warn("Set Environ Variable 'MASTER_PORT'='29500'")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def load_models(opt):
    # load nvae pretrained cifar10
    checkpoint = torch.load(opt.nvae_path, map_location='cpu')

    # get and update args
    args = checkpoint['args']
    args.num_mixture_dec = 10
    args.batch_size = opt.batch_size

    # init model and load
    arch_instance = get_arch_cells(args.arch_instance)
    nvae = AutoEncoder(args, None, arch_instance)
    nvae.load_state_dict(checkpoint['state_dict'], strict=False)
    nvae = ddp(nvae.to(opt.rank), device_ids=[opt.rank]).eval()

    # get size of each chunk
    # this is latent_size * latent_resolution ** 2
    # repeat it based on num groups per scale (latent resolution)
    chunk_size = [nvae.module.num_latent_per_group * (opt.img_size // (2 ** (i + 1))) ** 2
                  for i in range(nvae.module.num_latent_scales)]
    chunk_size.reverse()
    chunk_size = [s for s in chunk_size for _ in range(nvae.module.num_groups_per_scale)]

    # Walker
    nn_walker = NonlinearWalker(n_chunks=sum(nvae.module.groups_per_scale), chunk_size=chunk_size,
                                to_perturb=opt.chunks_to_perturb, init_weights=True)
    nn_walker = ddp(nn_walker.to(opt.rank), device_ids=[opt.rank])

    # Attacked CNN
    os.environ["TORCH_HOME"] = opt.torch_home
    resnet32 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True)
    resnet32 = ddp(resnet32.to(opt.rank), device_ids=[opt.rank]).eval()

    # Test CNN
    vgg16 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
    vgg16 = ddp(vgg16.to(opt.rank), device_ids=[opt.rank]).eval()

    return nvae, nn_walker, resnet32, vgg16


def prepare_data(opt):
    train_dataset = CIFAR10(root=opt.data_dir, train=True, transform=transforms.ToTensor(), download=True)
    train_sampler = DistributedSampler(train_dataset, num_replicas=opt.world_size, rank=opt.rank,
                                       shuffle=True, drop_last=False)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, pin_memory=False, num_workers=0,
                                  drop_last=False, sampler=train_sampler)

    test_dataset = CIFAR10(root=opt.data_dir, train=False, transform=transforms.ToTensor(), download=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=opt.world_size, rank=opt.rank,
                                      shuffle=False, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, pin_memory=False, num_workers=0,
                                 drop_last=False, sampler=test_sampler)

    return train_dataloader, test_dataloader


def train_step(opt, batch, cnn_preprocess, optimizer, models):
    walker, cnn, nvae = models

    optimizer.zero_grad()

    # get x batch, chunks and gt (according to cnn)
    x, _ = batch
    x = x.to(opt.rank)
    b, c, h, w = x.shape

    with torch.no_grad():
        # prediction according to attacked CNN
        top2_labels = torch.topk(cnn.module(cnn_preprocess(x)), k=2, dim=1).indices
        chunks, _ = nvae.module.encode(x)

    # get adversarial chunks
    adversarial_chunks = []
    for i, chunk in enumerate(chunks):
        adv_c = walker.module(chunk, i)
        r = int(math.sqrt(adv_c.shape[1] // nvae.module.num_latent_per_group))
        adversarial_chunks.append(adv_c.view(b, nvae.module.num_latent_per_group, r, r))

    # decode to get adversarial images (need grad for backprop)
    nvae_logits = nvae.module.decode(adversarial_chunks)
    adversarial_samples = nvae.module.decoder_output(nvae_logits).sample()

    # obtain cnn scores to minimize
    atk_preds = torch.softmax(cnn.module(cnn_preprocess(adversarial_samples)), dim=1)
    best_scores = atk_preds[torch.arange(b), top2_labels[:, 0]]

    # LOSSES
    # minimize best and maximize second best (both are bounded in 0__1 range)
    second_best_scores = atk_preds[torch.arange(b), top2_labels[:, 1]]
    loss_scores = (- torch.log10(1. - 0.9 * best_scores)) # - torch.log10(second_best_scores * 0.9 + 0.1)) * 0.5

    # loss_recon = (1. - ssim(adversarial_samples, x, reduction='none')) / 0.15  # bound at 0.85

    mse = torch.cdist(x.view(b, -1), adversarial_samples.view(b, -1)).diag().view(-1, 1)
    loss_recon = torch.mean(mse, dim=1) / 6.0  # L2 loss with bound 1.5

    # final Loss
    loss = torch.mean(loss_recon + loss_scores)  # both have max = 1

    loss.backward()
    optimizer.step()

    return loss_scores, loss_recon


def validation_step(opt, batch, preprocess_cnn, models):
    walker, attacked_cnn, tested_cnn, nvae = models

    # get x batch, chunks and gt (according to cnn)
    x, y = batch
    x = x.to(opt.rank)
    gt = y.to(opt.rank)
    b, _, _, _ = x.shape

    with torch.no_grad():
        # get gt prediction for CNNs
        clean_labels_atk = torch.argmax(attacked_cnn.module(preprocess_cnn(x)), dim=1)
        clean_labels_tst = torch.argmax(tested_cnn.module(preprocess_cnn(x)), dim=1)

        # obtain adversaries
        chunks, _ = nvae.module.encode(x)
        adversarial_chunks = []
        for i, c in enumerate(chunks):
            adv_c = walker.module(c, i)
            r = int(math.sqrt(adv_c.shape[1] // nvae.module.num_latent_per_group))
            adversarial_chunks.append(adv_c.view(b, nvae.module.num_latent_per_group, r, r))
        nvae_logits = nvae.module.decode(adversarial_chunks)
        adversarial_samples = nvae.module.decoder_output(nvae_logits).sample()

        # get predictions for adversaries
        adv_labels_atk = torch.argmax(attacked_cnn.module(preprocess_cnn(adversarial_samples)), dim=1)
        adv_labels_tst = torch.argmax(tested_cnn.module(preprocess_cnn(adversarial_samples)), dim=1)

    # send preds (B, 1) to rank 0
    return torch.cat([gt.unsqueeze(1), clean_labels_atk.unsqueeze(1), adv_labels_atk.unsqueeze(1),
                      clean_labels_tst.unsqueeze(1), adv_labels_tst.unsqueeze(1)], dim=1), adversarial_samples


def main(rank, world_size, opt):
    # just to pass only opt as argument to methods
    opt.rank = rank
    opt.world_size = world_size

    is_master = opt.rank == 0

    torch.cuda.set_device(opt.rank)
    setup(opt.rank, opt.world_size)

    # init models, dataset, criterion and optimizer
    nvae, walker, attacked_cnn, test_cnn = load_models(opt)
    train_dataloader, test_dataloader = prepare_data(opt)
    optimizer = optim.Adam(walker.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)

    cnn_preprocess = Normalize(mean=torch.tensor([0.507, 0.4865, 0.4409], device=opt.rank),
                               std=torch.tensor([0.2673, 0.2564, 0.2761], device=opt.rank))

    # train loss values for final plot (mean per each epoch)
    train_scores_loss_epoch = []
    train_recons_loss_epoch = []

    # accuracies epoch values for final plot
    atk_cnn_clean_acc_epoch = []
    atk_cnn_adv_acc_epoch = []
    tst_cnn_clean_acc_epoch = []
    tst_cnn_adv_acc_epoch = []

    for epoch in range(opt.n_epochs):

        dist.barrier()
        walker = walker.train()

        if is_master:
            print(f'Epoch: {epoch} / {opt.n_epochs}')

        # reset each epoch
        train_scores_loss_iter = []
        train_recons_loss_iter = []

        # Train loop:
        for batch in tqdm(train_dataloader):

            loss_scores, loss_recon = train_step(opt, batch, cnn_preprocess, optimizer, (walker, attacked_cnn, nvae))

            # at each training iter: pass loss terms of every image to rank 0
            if not is_master:
                dist.gather(tensor=loss_scores.detach(), dst=0)
                dist.gather(tensor=loss_recon.detach(), dst=0)
            else:

                collected_scores = [torch.zeros_like(loss_scores) for _ in range(world_size)]
                collected_recon = [torch.zeros_like(loss_recon) for _ in range(world_size)]

                dist.gather(gather_list=collected_scores, tensor=loss_scores.detach())
                dist.gather(gather_list=collected_recon, tensor=loss_recon.detach())

                # get full Batch mean losses for iteration.
                iter_score_loss = torch.mean(torch.cat(collected_scores, dim=0)).item()
                iter_recon_loss = torch.mean(torch.cat(collected_recon, dim=0)).item()

                if math.isnan(iter_score_loss) or math.isnan(iter_recon_loss):
                    print(f'NAN VALUE FOUND: scores = {iter_score_loss} ; recons  = {iter_recon_loss}')
                    print(f'LAST score losses: {train_scores_loss_iter}')
                    print(f'LAST recon losses: {train_recons_loss_iter}')

                # save every loss value for final plot.
                train_scores_loss_iter.append(iter_score_loss)
                train_recons_loss_iter.append(iter_recon_loss)

        if is_master:

            mean_score_loss = sum(train_scores_loss_iter) / len(train_scores_loss_iter)
            mean_recon_loss = sum(train_recons_loss_iter) / len(train_recons_loss_iter)

            train_scores_loss_epoch.append(mean_score_loss)
            train_recons_loss_epoch.append(mean_recon_loss)

            print(f'last loss value (scores): {mean_score_loss:.3f}')
            print(f'last loss value (recons): {mean_recon_loss:.3f}')
            print(f'Training Epoch Done. Validating now!')

        # sync all before test step
        dist.barrier()
        # ############################################################################################################

        walker = walker.eval()

        # label predictions (for computing epoch/accuracy)
        epoch_gt = torch.empty((0, 1), device=opt.rank)
        epoch_clean_atk = torch.empty((0, 1), device=opt.rank)
        epoch_adv_atk = torch.empty((0, 1), device=opt.rank)
        epoch_clean_tst = torch.empty((0, 1), device=opt.rank)
        epoch_adv_tst = torch.empty((0, 1), device=opt.rank)

        # Validation loop:
        for batch_index, batch in enumerate(tqdm(test_dataloader)):

            b, _, _, _ = batch[0].shape

            step_preds, adv_samples = validation_step(opt, batch, cnn_preprocess,
                                                      (walker, attacked_cnn, test_cnn, nvae))

            if not is_master:
                dist.gather(dst=0, tensor=step_preds)
            else:
                all_step_preds = [torch.zeros((b, 5), dtype=torch.int64).to(opt.rank) for _ in range(world_size)]
                dist.gather(gather_list=all_step_preds, tensor=step_preds)

                # (GLOBAL BS, 5)
                all_step_preds = torch.cat(all_step_preds, dim=0)

                # append iter predictions to epoch predictions
                epoch_gt = torch.cat([epoch_gt, all_step_preds[:, 0].unsqueeze(1)])
                epoch_clean_atk = torch.cat([epoch_clean_atk, all_step_preds[:, 1].unsqueeze(1)])
                epoch_adv_atk = torch.cat([epoch_adv_atk, all_step_preds[:, 2].unsqueeze(1)])
                epoch_clean_tst = torch.cat([epoch_clean_tst, all_step_preds[:, 3].unsqueeze(1)])
                epoch_adv_tst = torch.cat([epoch_adv_tst, all_step_preds[:, 4].unsqueeze(1)])

            # save images across epochs
            if is_master and batch_index == 0:
                l2_val = torch.mean(torch.cdist(batch[0].to(opt.rank) .view(b, -1), adv_samples.view(b, -1), p=2).diag()).item()
                ssim_val = ssim(batch[0].to(opt.rank), adv_samples).item()

                plt.imshow(
                    make_grid(
                        torch.cat([batch[0][:8].to(opt.rank), adv_samples[:8]], dim=0)
                    ).permute(1, 2, 0).cpu().numpy())
                plt.title(f'L2: {l2_val:.4f} - SSIM: {ssim_val:.3f}')
                plt.savefig(f'{opt.plots_folder}/samples_ep{epoch:02d}.png')
                plt.close()

        # compute epoch accuracy, add for final plot and save model.
        if is_master:
            num_classes = int(torch.max(epoch_gt).item()) + 1
            clean_acc_atk = accuracy(epoch_clean_atk, epoch_gt, task='multiclass', num_classes=num_classes)
            adv_acc_atk = accuracy(epoch_adv_atk, epoch_gt, task='multiclass', num_classes=num_classes)
            clean_acc_tst = accuracy(epoch_clean_tst, epoch_gt, task='multiclass', num_classes=num_classes)
            adv_acc_tst = accuracy(epoch_adv_tst, epoch_gt, task='multiclass', num_classes=num_classes)

            print(f'Clean CNN Accuracy: {clean_acc_atk} - Adversarial CNN Accuracy {adv_acc_atk}')
            print('#' * 30)

            atk_cnn_clean_acc_epoch.append(clean_acc_atk.item())
            atk_cnn_adv_acc_epoch.append(adv_acc_atk.item())
            tst_cnn_clean_acc_epoch.append(clean_acc_tst.item())
            tst_cnn_adv_acc_epoch.append(adv_acc_tst.item())

            # saving model
            torch.save(walker.state_dict(), f'{opt.ckpt_folder}/ep_{epoch:02d}.pt')

    # save final plots
    if is_master:

        # Losses Plot
        fig, [scores, recons] = plt.subplots(1, 2, figsize=(12, 8))
        scores.plot(train_scores_loss_epoch)
        scores.set_title('Training Loss - Scores')
        scores.set_xlabel('epochs')
        scores.set_ylabel('loss')

        min_v, max_v = min(train_scores_loss_epoch), max(train_scores_loss_epoch)
        step = (max_v - min_v) / 20
        y_values = list(np.arange(min_v, max_v + step, step))
        scores.set_yticks(y_values, [f'{n:.3f}' for n in y_values])

        recons.plot(train_recons_loss_epoch)
        recons.set_title('Training Loss - Recons')
        recons.set_xlabel('epochs')
        recons.set_ylabel('loss')
        min_v, max_v = min(train_recons_loss_epoch), max(train_recons_loss_epoch)
        step = (max_v - min_v) / 20
        y_values = list(np.arange(min_v, max_v + step, step))
        recons.set_yticks(y_values, [f'{n:.3f}' for n in y_values])

        plt.savefig(f'{opt.plots_folder}/train_loss.png')
        plt.close()

        # Test accuracy plot
        fig, [atk, tst] = plt.subplots(1, 2, figsize=(12, 8))
        atk.plot(atk_cnn_clean_acc_epoch, c='blue')
        atk.plot(atk_cnn_adv_acc_epoch, c='orange')
        atk.set_title('Resnet 32 - Attacked')
        atk.set_xlabel('Epochs')
        atk.set_ylabel('Accuracy (Top-1)')

        blue, = tst.plot(tst_cnn_clean_acc_epoch, c='blue')
        orange, = tst.plot(tst_cnn_adv_acc_epoch, c='orange')
        tst.set_title('VGG 16 - Tested')
        tst.set_xlabel('Epochs')
        tst.set_ylabel('Accuracy (Top-1)')

        plt.legend([blue, orange], ['clean accuracy', 'adversarial accuracy'])
        plt.savefig(f'{opt.plots_folder}/test_accuracies.png')
        plt.close()

    dist.destroy_process_group()


if __name__ == '__main__':

    w_size = torch.cuda.device_count()
    print(f'Using {w_size} gpus for training model')

    cudnn.benchmark = True

    try:
        mp.spawn(main, args=(w_size, init_run()), nprocs=w_size)
    except KeyboardInterrupt as k_int:
        print('Detected Keyboard Interrupt! Attempting to kill all processes')
        dist.destroy_process_group()
