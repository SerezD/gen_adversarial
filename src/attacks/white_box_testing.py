import os
import argparse
import warnings

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torchvision.transforms
from torch.nn.parallel import DistributedDataParallel as ddp

import math
from kornia.enhance import Normalize
import matplotlib.pyplot as plt

from torch.utils.data import DistributedSampler, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
from torchmetrics import Accuracy
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm

from src.attacks.latent_walkers import NonlinearWalker

from src.NVAE.original.model import AutoEncoder
from src.NVAE.original.utils import get_arch_cells


def init_run():
    parser = argparse.ArgumentParser('train a non linear latent walker for a white box attack setting.')

    parser.add_argument('--nvae_path', type=str, help='pretrained NVAE path',
                        default='/media/dserez/runs/NVAE/cifar10/best/ours_weighted.pt')
    parser.add_argument('--torch_home', type=str,
                        default='/media/dserez/runs/classification/cifar10/')
    parser.add_argument('--data_dir', type=str,
                        default='/media/dserez/code/adversarial/')

    parser.add_argument('--n_epochs', type=int, default=10)
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
                                init_weights=True)
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
    train_dataset = CIFAR10(root=opt.data_dir, train=True, transform=torchvision.transforms.ToTensor(), download=True)
    train_sampler = DistributedSampler(train_dataset, num_replicas=opt.world_size, rank=opt.rank,
                                       shuffle=True, drop_last=False)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, pin_memory=False, num_workers=0,
                                  drop_last=False, sampler=train_sampler)

    test_dataset = CIFAR10(root=opt.data_dir, train=False, transform=torchvision.transforms.ToTensor(), download=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=opt.world_size, rank=opt.rank,
                                      shuffle=False, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, pin_memory=False, num_workers=0,
                                 drop_last=False, sampler=test_sampler)

    return train_dataloader, test_dataloader


def main(rank, world_size, opt):

    # just to pass only opt as argument to methods
    opt.rank = rank
    opt.world_size = world_size

    torch.cuda.set_device(opt.rank)
    setup(opt.rank, opt.world_size)

    # init models, dataset, criterion and optimizer
    nvae, walker, attacked_cnn, test_cnn = load_models(opt)
    train_dataloader, test_dataloader = prepare_data(opt)
    criterion_reconstruction = StructuralSimilarityIndexMeasure(reduction="none").to(opt.rank)
    optimizer = optim.Adam(walker.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)

    # accuracy for validation
    accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1).to(opt.rank)
    best_accuracy = 1.  # monitor to save model
    preprocess = Normalize(mean=torch.tensor([0.507, 0.4865, 0.4409], device=opt.rank),
                           std=torch.tensor([0.2673, 0.2564, 0.2761], device=opt.rank))

    # values to save for plotting
    if opt.rank == 0:

        # accuracies
        atk_cnn_clean_acc = []
        atk_cnn_adv_acc = []
        tst_cnn_clean_acc = []
        tst_cnn_adv_acc = []

        # all losses (per iter)
        scores_losses = []
        recons_losses = []

    for epoch in range(opt.n_epochs):

        if opt.rank == 0:
            print(f'Epoch: {epoch} / {opt.n_epochs}')

        # TRAIN LOOP
        for batch_index, batch in enumerate(tqdm(train_dataloader)):

            # get x batch, chunks and gt (according to cnn)
            x, _ = batch
            x = x.to(opt.rank)
            b, _, size, size = x.shape

            with torch.no_grad():
                gt = torch.argmax(attacked_cnn.module(preprocess(x)), dim=1)
                chunks = nvae.module.encode(x)

            optimizer.zero_grad()

            # get adversarial chunks
            adversarial_chunks = []
            for i, c in enumerate(chunks):
                adv_c = walker.module(c, i)
                r = int(math.sqrt(adv_c.shape[1] // nvae.module.num_latent_per_group))
                adversarial_chunks.append(adv_c.view(b, nvae.module.num_latent_per_group, r, r))

            # decode to get adversarial images (need grad for backprop)
            nvae_logits = nvae.module.decode(adversarial_chunks)
            adversarial_samples = nvae.module.decoder_output(nvae_logits).sample()

            # obtain cnn scores to minimize
            loss_scores = torch.softmax(attacked_cnn.module(adversarial_samples), dim=1)[torch.arange(b), gt]
            loss_recon = 1. - criterion_reconstruction(adversarial_samples, x)
            loss = torch.mean(loss_recon + loss_scores)
            loss.backward()
            optimizer.step()

        if opt.rank == 0:
            collected_scores = [torch.zeros_like(loss_scores) for _ in range(world_size)]
            dist.gather(gather_list=collected_scores, tensor=loss_scores)
            collected_recon = [torch.zeros_like(loss_recon) for _ in range(world_size)]
            dist.gather(gather_list=collected_recon, tensor=loss_recon)

        else:
            dist.gather(tensor=loss_scores, dst=0)
            dist.gather(tensor=loss_recon, dst=0)

        if opt.rank == 0:
            epoch_score = torch.mean(torch.cat(collected_scores, dim=0)).item()
            epoch_recon = torch.mean(torch.cat(collected_recon, dim=0)).item()

            print(f'mean Loss (scores): {epoch_score}')
            print(f'mean Loss (recons): {epoch_recon}')

            scores_losses.append(epoch_score)
            recons_losses.append(epoch_recon)

        # ###########################################################################################################
        if opt.rank == 0:
            print(f'Train Done. Validating now!')

        # sync all before test step
        dist.barrier()

        with torch.no_grad():

            walker = walker.eval()

            # load predictions to compute Accuracy at epoch end
            if opt.rank == 0:
                epoch_gt = torch.empty((0, 1), device=opt.rank)
                epoch_clean_atk = torch.empty((0, 1), device=opt.rank)
                epoch_adv_atk = torch.empty((0, 1), device=opt.rank)
                epoch_clean_tst = torch.empty((0, 1), device=opt.rank)
                epoch_adv_tst = torch.empty((0, 1), device=opt.rank)

            for batch_index, batch in enumerate(tqdm(test_dataloader)):

                # get x batch, chunks and gt (according to cnn)
                x, y = batch
                x = x.to(opt.rank)
                gt = y.to(opt.rank)
                b, _, size, size = x.shape

                clean_labels_atk = torch.argmax(attacked_cnn.module(preprocess(x)), dim=1)
                clean_labels_tst = torch.argmax(test_cnn.module(preprocess(x)), dim=1)

                # obtain adversaries
                chunks = nvae.module.encode(x)
                adversarial_chunks = []
                for i, c in enumerate(chunks):
                    adv_c = walker.module(c, i)
                    r = int(math.sqrt(adv_c.shape[1] // nvae.module.num_latent_per_group))
                    adversarial_chunks.append(adv_c.view(b, nvae.module.num_latent_per_group, r, r))
                nvae_logits = nvae.module.decode(adversarial_chunks)
                adversarial_samples = nvae.module.decoder_output(nvae_logits).sample()

                adv_labels_atk = torch.argmax(attacked_cnn.module(preprocess(adversarial_samples)), dim=1)
                adv_labels_tst = torch.argmax(test_cnn.module(preprocess(adversarial_samples)), dim=1)

                # send preds to rank 0
                tensor_final = torch.cat(
                                    [gt.unsqueeze(1),
                                     clean_labels_atk.unsqueeze(1),
                                     adv_labels_atk.unsqueeze(1),
                                     clean_labels_tst.unsqueeze(1),
                                     adv_labels_tst.unsqueeze(1)], dim=1)
                if opt.rank == 0:
                    all_preds = [torch.zeros((b, 5), dtype=torch.int64).to(opt.rank) for _ in range(world_size)]
                    dist.gather(gather_list=all_preds, tensor=tensor_final)
                else:
                    dist.gather(dst=0, tensor=tensor_final)

                # collect preds and append
                if opt.rank == 0:
                    all_preds = torch.cat(all_preds, dim=0)
                    epoch_gt = torch.cat([epoch_gt, all_preds[:, 0].unsqueeze(1)])
                    epoch_clean_atk = torch.cat([epoch_clean_atk, all_preds[:, 1].unsqueeze(1)])
                    epoch_adv_atk = torch.cat([epoch_adv_atk, all_preds[:, 2].unsqueeze(1)])
                    epoch_clean_tst = torch.cat([epoch_clean_tst, all_preds[:, 3].unsqueeze(1)])
                    epoch_adv_tst = torch.cat([epoch_adv_tst, all_preds[:, 4].unsqueeze(1)])

                # save images across epochs
                if opt.rank == 0 and batch_index == 0:
                    plt.imshow(
                        make_grid(
                            torch.cat([x[:8], adversarial_samples[:8]], dim=0)
                        ).permute(1, 2, 0).detach().cpu().numpy())
                    plt.savefig(f'{opt.plots_folder}/samples_ep{epoch:.2f}.png')
                    plt.close()

            if opt.rank == 0:

                clean_acc_atk = accuracy(epoch_clean_atk, epoch_gt)
                adv_acc_atk = accuracy(epoch_adv_atk, epoch_gt)
                clean_acc_tst = accuracy(epoch_clean_tst, epoch_gt)
                adv_acc_tst = accuracy(epoch_adv_tst, epoch_gt)

                atk_cnn_clean_acc.append(clean_acc_atk.item())
                atk_cnn_adv_acc.append(adv_acc_atk.item())
                tst_cnn_clean_acc.append(clean_acc_tst.item())
                tst_cnn_adv_acc.append(adv_acc_tst.item())

                if adv_acc_atk < best_accuracy:
                    #best_accuracy = adv_acc_atk

                    # saving model
                    torch.save(walker.state_dict(), f'{opt.ckpt_folder}/ep_{epoch:.2f}')

            walker = walker.train()

        # sync all before new epoch
        dist.barrier()

    # save final plots
    if opt.rank == 0:

        # Losses Plot
        fig, [scores, recons] = plt.subplots(1, 2, figsize=(12, 8))
        scores.plot(scores_losses)
        scores.set_title('Training Loss - Scores')
        scores.set_xlabel('epochs')
        scores.set_ylabel('loss')

        recons.plot(recons_losses)
        recons.set_title('Training Loss - Recons')
        recons.set_xlabel('epochs')
        recons.set_ylabel('loss')

        plt.savefig(f'{opt.plots_folder}/train_loss.png')
        plt.close()

        # Test accuracy plot
        fig, [atk, tst] = plt.subplots(1, 2, figsize=(12, 8))
        atk.plot(atk_cnn_clean_acc, c='blue')
        atk.plot(atk_cnn_adv_acc, c='orange')
        atk.set_title('Resnet 32 - Attacked')
        atk.set_xlabel('Epochs')
        atk.set_ylabel('Accuracy (Top-1)')

        blue, = tst.plot(tst_cnn_clean_acc, c='blue')
        orange, = tst.plot(tst_cnn_adv_acc, c='orange')
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

    mp.spawn(main, args=(w_size, init_run()), nprocs=w_size)
