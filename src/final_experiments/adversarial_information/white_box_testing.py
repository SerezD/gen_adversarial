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
from torch.nn.parallel import DistributedDataParallel as ddp
from torch.utils.data import DistributedSampler, DataLoader

from torchvision.utils import make_grid
from torchmetrics.functional import accuracy
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

from tqdm import tqdm

from data.datasets import ImageLabelDataset
from src.NVAE.mine.distributions import DiscMixLogistic
from src.final_experiments.adversarial_information.latent_walkers import NonlinearWalker
from src.NVAE.mine.model import AutoEncoder

import warnings


def init_run():
    parser = argparse.ArgumentParser('train a non linear latent walker for a white box attack setting.')

    parser.add_argument('--nvae_path', type=str, help='pretrained NVAE path',
                        default='/media/dserez/runs/NVAE/cifar10/ours/replica/no_kl_cost.pt')
    parser.add_argument('--torch_home', type=str,
                        default='/media/dserez/runs/adversarial/CNNs/')
    parser.add_argument('--data_dir', type=str,
                        default='/media/dserez/datasets/cifar10/')

    parser.add_argument('--chunks_to_perturb', type=int, default=None, nargs='+',
                        help='indices of chunks to perturb, starting from 0')
    parser.add_argument('--l2_bound', type=float, default=0.5, help='L2 norm bound')

    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
    parser.add_argument('--attacked_cnn', type=str, default='resnet', choices=['resnet', 'vgg'])

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


def setup(opt):
    if 'MASTER_ADDR' not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        if opt.rank == 0:
            warnings.warn("Set Environ Variable 'MASTER_ADDR'='localhost'")

    if 'MASTER_PORT' not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
        if opt.rank == 0:
            warnings.warn("Set Environ Variable 'MASTER_PORT'='29500'")

    dist.init_process_group("nccl", rank=opt.rank, world_size=opt.world_size)

    # ensures that weight initializations are all the same
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)


def load_models(opt):
    def get_model_conf(filepath: str):
        import yaml

        # load params
        with open(filepath, 'r', encoding='utf-8') as stream:
            params = yaml.safe_load(stream)

        return params

    # load pretrained nvae
    conf_file = '/'.join(opt.nvae_path.split('/')[:-1]) + '/conf.yaml'
    config = get_model_conf(conf_file)

    nvae = AutoEncoder(config['autoencoder'], config['resolution'])
    checkpoint = torch.load(opt.nvae_path, map_location='cpu')

    nvae.load_state_dict(checkpoint['state_dict'])
    nvae.cuda().eval()
    nvae = ddp(nvae.to(opt.rank), device_ids=[opt.rank]).eval()

    # get size of each chunk
    # this is latent_size * latent_resolution ** 2
    # repeat it based on num groups per scale (latent resolution)
    chunk_size = [nvae.module.num_latent_per_group * (opt.img_size // (2 ** (i + 1))) ** 2
                  for i in range(nvae.module.num_scales)]
    chunk_size.reverse()
    chunk_size_final = []
    for n, s in zip(nvae.module.groups_per_scale, chunk_size):
        for _ in range(n):
            chunk_size_final.append(s)

    # Walker
    nn_walker = NonlinearWalker(n_chunks=sum(nvae.module.groups_per_scale), chunk_size=chunk_size_final,
                                to_perturb=opt.chunks_to_perturb, init_weights=True)
    nn_walker = ddp(nn_walker.to(opt.rank), device_ids=[opt.rank])

    # Attacked CNN
    os.environ["TORCH_HOME"] = opt.torch_home
    if opt.attacked_cnn == 'resnet':
        cnn = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True)
    else:
        cnn = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
    cnn = ddp(cnn.to(opt.rank), device_ids=[opt.rank]).eval()

    return nvae, nn_walker, cnn


def prepare_data(opt):

    image_size = opt.img_size
    batch_size = opt.batch_size // opt.world_size
    seed = int(opt.seed)
    is_distributed = opt.world_size > 1

    # create dataset objects
    train_dataset = ImageLabelDataset(folder=f'{opt.data_dir}/train', image_size=image_size, ffcv=False)
    val_dataset = ImageLabelDataset(folder=f'{opt.data_dir}/validation', image_size=image_size, ffcv=False)

    # create sampler
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=opt.world_size, rank=opt.rank,
                                           shuffle=True, drop_last=True, seed=seed)
        val_sampler = DistributedSampler(val_dataset, num_replicas=opt.world_size, rank=opt.rank,
                                         shuffle=False, drop_last=False, seed=seed)
    else:
        # use default sampler
        train_sampler, val_sampler = None, None

    # final loader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=0,
                                  sampler=train_sampler, drop_last=train_sampler is None, shuffle=train_sampler is None)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=0, sampler=val_sampler)

    if opt.rank == 0:
        print(f"[INFO] final batch size per device: {batch_size}")

    return train_dataloader, val_dataloader


def train_step(opt, batch, cnn_preprocess, optimizer, models):

    walker, cnn, nvae = models

    optimizer.zero_grad()

    # get x batch, chunks and gt (according to cnn)
    x, _ = batch
    x = x.to(opt.rank)
    b, c, h, w = x.shape

    with torch.no_grad():
        # prediction according to attacked CNN
        chunks = nvae.module.encode(x, deterministic=True)
        logits = nvae.module.decode(chunks)
        recons = DiscMixLogistic(logits, img_channels=3, num_bits=8).mean()
        clean_labels = torch.argmax(cnn.module(cnn_preprocess(recons)), dim=1)

    # get adversarial chunks
    adversarial_chunks = []
    for i, chunk in enumerate(chunks):
        adv_c = walker.module(chunk.view(b, -1), i)
        r = int(math.sqrt(adv_c.shape[1] // nvae.module.num_latent_per_group))
        adversarial_chunks.append(adv_c.view(b, nvae.module.num_latent_per_group, r, r))

    # decode to get adversarial images (need grad for backprop)
    nvae_logits = nvae.module.decode(adversarial_chunks)
    adversarial_samples = DiscMixLogistic(nvae_logits, img_channels=3, num_bits=8).mean()

    # force adversarial_samples to bound w.r.t reconstructions
    delta = (adversarial_samples - recons).view(b, -1)
    delta_norm = delta.norm(p=2, dim=1, keepdim=True)
    eps = torch.ones_like(delta_norm) * float(opt.l2_bound)

    # get delta in bound
    scaling_factor = torch.where(delta_norm <= eps, eps, delta_norm)
    delta = delta * eps / scaling_factor

    adversarial_samples = recons + delta.view(b, c, h, w)

    # obtain cnn scores to minimize
    atk_preds = torch.softmax(cnn.module(cnn_preprocess(adversarial_samples)), dim=1)
    best_scores = atk_preds[torch.arange(b), clean_labels]

    # LOSSES
    # minimize best
    loss_scores = -torch.log10(1. - 0.9 * best_scores)
    l2_dist = (recons - adversarial_samples).view(b, -1).norm(p=2, dim=1)

    # final Loss
    loss = torch.mean(l2_dist + loss_scores * float(opt.l2_bound))  # both have max l2_bound

    loss.backward()
    optimizer.step()

    return loss_scores.detach(), l2_dist.detach()


def validation_step(opt, batch, preprocess_cnn, models):

    walker, attacked_cnn, nvae = models

    # get x batch, chunks and gt (according to cnn)
    x, y = batch
    x = x.to(opt.rank)
    gt = y.to(opt.rank)
    b, _, _, _ = x.shape

    with torch.no_grad():

        # get gt prediction for CNNs on reconstructions
        chunks = nvae.module.encode(x, deterministic=True)
        reconstructions = DiscMixLogistic(nvae.module.decode(chunks), img_channels=3, num_bits=8).mean()
        clean_labels_atk = torch.argmax(attacked_cnn.module(preprocess_cnn(reconstructions)), dim=1)

        # obtain adversaries
        adversarial_chunks = []
        for i, c in enumerate(chunks):
            adv_c = walker.module(c.view(b, -1), i)
            r = int(math.sqrt(adv_c.shape[1] // nvae.module.num_latent_per_group))
            adversarial_chunks.append(adv_c.view(b, nvae.module.num_latent_per_group, r, r))

        nvae_logits = nvae.module.decode(adversarial_chunks)
        adversarial_samples = DiscMixLogistic(nvae_logits, img_channels=3, num_bits=8).mean()

        # get predictions for adversaries
        adv_labels_atk = torch.argmax(attacked_cnn.module(preprocess_cnn(adversarial_samples)), dim=1)

    # send preds (B, 1) to rank 0
    return torch.cat([gt.unsqueeze(1), clean_labels_atk.unsqueeze(1), adv_labels_atk.unsqueeze(1)], dim=1), adversarial_samples


def main(rank, world_size, opt):

    # just to pass only opt as argument to methods
    opt.rank = rank
    opt.world_size = world_size

    is_master = opt.rank == 0

    torch.cuda.set_device(opt.rank)
    setup(opt)

    # init models, dataset, criterion and optimizer
    nvae, walker, attacked_cnn = load_models(opt)
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

    for epoch in range(opt.n_epochs):

        if world_size > 1:
            train_dataloader.sampler.set_epoch(epoch)

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
            dist.all_reduce(loss_scores, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_recon, op=dist.ReduceOp.SUM)

            if is_master:

                # get full Batch mean losses for iteration.
                iter_score_loss = torch.mean(loss_scores / opt.world_size).item()
                iter_recon_loss = torch.mean(loss_recon / opt.world_size).item()

                if math.isnan(iter_score_loss) or math.isnan(iter_recon_loss):
                    print(f'NAN VALUE FOUND: scores = {iter_score_loss} ; recons  = {iter_recon_loss}')
                    print(f'LAST score losses: {train_scores_loss_iter}')
                    print(f'LAST recon losses: {train_recons_loss_iter}')

                # save every loss value for final plot.
                train_scores_loss_iter.append(iter_score_loss)
                train_recons_loss_iter.append(iter_recon_loss)

        # epoch end
        if is_master:

            mean_score_loss = sum(train_scores_loss_iter) / len(train_scores_loss_iter)
            mean_recon_loss = sum(train_recons_loss_iter) / len(train_recons_loss_iter)

            train_scores_loss_epoch.append(mean_score_loss)
            train_recons_loss_epoch.append(mean_recon_loss)

            print(f'train loss (scores): {mean_score_loss:.3f}')
            print(f'train loss (recons): {mean_recon_loss:.3f}')
            print(f'Training Epoch Done. Validating now!')

        # sync all before test step
        dist.barrier()
        # ############################################################################################################

        walker = walker.eval()

        # label predictions (for computing epoch/accuracy)
        epoch_gt = torch.empty((0, 1), device=opt.rank)
        epoch_clean_atk = torch.empty((0, 1), device=opt.rank)
        epoch_adv_atk = torch.empty((0, 1), device=opt.rank)

        # Validation loop:
        for batch_index, batch in enumerate(tqdm(test_dataloader)):

            b, _, _, _ = batch[0].shape

            step_preds, adv_samples = validation_step(opt, batch, cnn_preprocess, (walker, attacked_cnn, nvae))

            dist.all_reduce(step_preds, op=dist.ReduceOp.SUM)
            step_preds = step_preds / opt.world_size

            if is_master:

                # append iter predictions to epoch predictions
                epoch_gt = torch.cat([epoch_gt, step_preds[:, 0].unsqueeze(1)])
                epoch_clean_atk = torch.cat([epoch_clean_atk, step_preds[:, 1].unsqueeze(1)])
                epoch_adv_atk = torch.cat([epoch_adv_atk, step_preds[:, 2].unsqueeze(1)])

            # save images across epochs
            if is_master and batch_index == 0:

                with torch.no_grad():
                    batch_0 = batch[0].to(opt.rank).view(b, -1)
                    adv_0 = adv_samples.view(b, -1)
                    rec_0 = nvae.module.autoencode(batch[0].to(opt.rank), deterministic=True)
                    rec_0 = DiscMixLogistic(rec_0, img_channels=3, num_bits=8).mean().view(b, -1)
                    l2_val = torch.mean(torch.cdist(batch_0, adv_0, p=2).diag()).item()
                    l2_recons_val = torch.mean(torch.cdist(rec_0, adv_0, p=2).diag()).item()

                plt.imshow(
                    make_grid(
                        torch.cat([batch[0][:8].to(opt.rank), adv_samples[:8]], dim=0)
                    ).permute(1, 2, 0).cpu().numpy())
                plt.title(f'L2 w.r.t original: {l2_val:.4f} - L2 w.r.t recons: {l2_recons_val:.3f}')
                plt.savefig(f'{opt.plots_folder}/samples_ep{epoch:02d}.png')
                plt.close()

        # compute epoch accuracy, add for final plot and save model.
        if is_master:
            num_classes = int(torch.max(epoch_gt).item()) + 1
            clean_acc_atk = accuracy(epoch_clean_atk, epoch_gt, task='multiclass', num_classes=num_classes)
            adv_acc_atk = accuracy(epoch_adv_atk, epoch_gt, task='multiclass', num_classes=num_classes)

            print(f'Clean CNN Accuracy: {clean_acc_atk} - Adversarial CNN Accuracy {adv_acc_atk}')
            print('#' * 30)

            atk_cnn_clean_acc_epoch.append(clean_acc_atk.item())
            atk_cnn_adv_acc_epoch.append(adv_acc_atk.item())

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
        plt.plot(atk_cnn_clean_acc_epoch, c='blue', label='Clean Accuracy')
        plt.plot(atk_cnn_adv_acc_epoch, c='orange', label='Adversarial Accuracy')
        plt.title(f'{opt.attacked_cnn} - Epochs Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (Top-1)')

        plt.legend()
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
