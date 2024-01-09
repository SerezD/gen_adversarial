import argparse
import math
import os
from typing import Any
import wandb

import numpy as np

import torch
from scheduling_utils.schedulers_cpp import LinearCosineScheduler, CosineScheduler
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
from pytorch_model_summary import summary
from torchvision.utils import make_grid
from tqdm import tqdm

from data.datasets import ImageDataset
from src.NVAE.mine.distributions import DiscMixLogistic
from src.NVAE.mine.model import AutoEncoder
from src.NVAE.mine.utils import kl_balancer


def parse_args():

    parser = argparse.ArgumentParser('NVAE training')

    parser.add_argument('--run_name', type=str, required=True,
                        help='unique name of the training run')

    parser.add_argument('--conf_file', type=str, required=True,
                        help='.yaml configuration file')

    parser.add_argument('--data_path', type=str, required=True,
                        help='directory of dataset, contains a "train" and "validation" subdirectories')

    parser.add_argument('--checkpoint_base_path', type=str, default='./runs/',
                        help='directory where checkpoints are saved')

    parser.add_argument('--resume_from', type=str, default=None,
                        help='if specified, resume training from this checkpoint')

    parser.add_argument('--logging', help='if passed, wandb logger is used', action='store_true')

    parser.add_argument('--wandb_project', type=str, help='project name for wandb logger', default='nvae')

    parser.add_argument('--wandb_id', type=str,
                        help='wandb id of the run. Useful for resuming logging of a model', default=None)

    args = parser.parse_args()

    # check data dir exists
    if not os.path.exists(f'{args.data_path}/train/') or not os.path.exists(f'{args.data_path}/validation/'):
        raise FileNotFoundError(f'{args.data_path}/train or {args.data_path}/validation does not exist')

    # check checkpoint file exists
    if args.resume_from is not None and not os.path.exists(f'{args.resume_from}/'):
        raise FileNotFoundError(f'could not find the specified checkpoint: {args.resume_from}')

    # create checkpoint out directory
    if not os.path.exists(f'{args.checkpoint_base_path}/{args.run_name}'):
        os.makedirs(f'{args.checkpoint_base_path}/{args.run_name}')

    return args


def get_model_conf(filepath: str):
    import yaml

    # load params
    with open(filepath, 'r', encoding='utf-8') as stream:
        params = yaml.safe_load(stream)

    return params


def setup(rank: int, world_size: int, train_conf: dict):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # ensures that weight initializations are all the same
    torch.manual_seed(train_conf['seed'])
    np.random.seed(train_conf['seed'])
    torch.cuda.manual_seed(train_conf['seed'])
    torch.cuda.manual_seed_all(train_conf['seed'])


def cleanup():
    dist.destroy_process_group()


def prepare_data(rank: int, world_size: int, data_dir: str, conf: dict):

    image_size = conf['resolution'][1]
    batch_size = conf['training']['cumulative_bs'] // world_size

    train_dataset = ImageDataset(folder=f'{data_dir}/train', image_size=image_size, ffcv=False)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                                       shuffle=True, drop_last=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=False, num_workers=0,
                                  drop_last=False, sampler=train_sampler)

    val_dataset = ImageDataset(folder=f'{data_dir}/validation', image_size=image_size, ffcv=False)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank,
                                     shuffle=False, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=False, num_workers=0,
                                drop_last=False, sampler=val_sampler)

    bpd_coefficient = 1. / np.log(2.) / (conf['resolution'][0] * conf['resolution'][1] * conf['resolution'][2])
    if rank == 0:
        print(f"[INFO] final batch size per device: {batch_size}")
        print(f"[INFO] Bit Per Dimension (bpd) coefficient: {bpd_coefficient}")

    return train_dataloader, val_dataloader, bpd_coefficient


def epoch_train(dataloader: DataLoader, model: AutoEncoder, optimizer: torch.optim.Optimizer, scheduler: Any,
                grad_scalar: GradScaler, kl_params: dict, sr_params: dict,
                total_training_steps: int, global_step: int, rank: int, world_size: int, run: wandb.run = None):
    """
    :param dataloader: train dataloader.
    :param model: model in training mode. Remember to pass ".module" with DDP.
    :param optimizer: optimizer object from torch.optim.Optimizer.
    :param scheduler: scheduler object from scheduling utils.
    :param grad_scalar: for AMP mode.
    :param kl_params: parameters for kl annealing coefficients.
    :param sr_params: spectral regularization parameters for lambda computation
    :param total_training_steps: total training steps on all epochs.
    :param global_step: used by scheduler.
    :param rank: for computing everything on the correct device.
    :param world_size: total number of devices
    :param run: wandb run object (if rank is 0).
    """

    epoch_loss = []
    epoch_recon_loss = []
    epoch_kl_loss = []
    epoch_spectral_loss = []
    epoch_bn_loss = []

    for step, x in enumerate(tqdm(dataloader)):

        x = x.to(f'cuda:{rank}')
        # b, c, h, w = x.shape
        # device = x.device

        # scheduler step
        lr = scheduler.step(global_step)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # use autocast for step (AMP)
        optimizer.zero_grad()
        with autocast():

            # forward pass
            logits, _, _, kl_all, _ = model(x)
            reconstructions = DiscMixLogistic(logits).log_prob(x)

            # reconstruction loss
            rec_loss = - torch.sum(reconstructions, dim=1)

            # compute kl weight (Appendix A of NVAE paper, linear warmup KL Term β)
            kl_weight = (global_step - float(kl_params["kl_const_portion"]) * total_training_steps)
            kl_weight /= (float(kl_params["kl_anneal_portion"]) * total_training_steps)
            kl_weight = max(min(1.0, kl_weight), float(kl_params["kl_const_coeff"]))

            # balance kl (Appendix A of NVAE paper, γ term on each scale)
            final_kl, kl_coefficients, _ = kl_balancer(kl_all, kl_weight, kl_balance=True, alpha=model.kl_alpha)

            # compute final loss
            loss = torch.mean(rec_loss + final_kl)

            # get spectral regularization coefficient λ
            if sr_params["weight_decay_norm_anneal"]:
                lambda_coeff = kl_weight * np.log(float(sr_params["weight_decay_norm"]))
                lambda_coeff += (1. - kl_weight) * np.log(float(sr_params["weight_decay_norm_init"]))
                lambda_coeff = np.exp(lambda_coeff)
            else:
                lambda_coeff = float(sr_params["weight_decay_norm"])

            # add spectral regularization and batch norm regularization terms to loss
            spectral_norm_term = model.spectral_norm_parallel()
            batch_norm_term = model. model.batch_norm_loss()
            loss += lambda_coeff * spectral_norm_term + lambda_coeff * batch_norm_term

        grad_scalar.scale(loss).backward()
        grad_scalar.step(optimizer)
        grad_scalar.update()

        # to log at each step
        if rank == 0:

            log_dict = {'lr': lr, 'KL beta': kl_weight, 'Lambda': lambda_coeff}
            for i, v in enumerate(kl_coefficients.cpu().numpy()):
                log_dict[f'KL gamma {i}'] = v

            run.log(log_dict, step=global_step)

        # save all loss terms to rank 0
        if not rank == 0:
            dist.gather(tensor=loss.detach(), dst=0)
            dist.gather(tensor=rec_loss.detach(), dst=0)
            dist.gather(tensor=final_kl.detach(), dst=0)
            dist.gather(tensor=spectral_norm_term.detach() * lambda_coeff, dst=0)
            dist.gather(tensor=batch_norm_term.detach() * lambda_coeff, dst=0)
        else:

            batch_loss = [torch.zeros_like(loss) for _ in range(world_size)]
            batch_recon_loss = [torch.zeros_like(rec_loss) for _ in range(world_size)]
            batch_kl_loss = [torch.zeros_like(final_kl) for _ in range(world_size)]
            batch_spectral_loss = [torch.zeros_like(spectral_norm_term) for _ in range(world_size)]
            batch_bn_loss = [torch.zeros_like(batch_norm_term) for _ in range(world_size)]

            dist.gather(gather_list=batch_loss, tensor=loss.detach())
            dist.gather(gather_list=batch_recon_loss, tensor=rec_loss.detach())
            dist.gather(gather_list=batch_kl_loss, tensor=final_kl.detach())
            dist.gather(gather_list=batch_spectral_loss, tensor=spectral_norm_term.detach() * lambda_coeff)
            dist.gather(gather_list=batch_bn_loss, tensor=batch_norm_term.detach() * lambda_coeff)

            # get full Batch mean losses for iteration.
            epoch_loss.append(torch.mean(torch.cat(batch_loss, dim=0)).item())
            epoch_recon_loss.append(torch.mean(torch.cat(batch_recon_loss, dim=0)).item())
            epoch_kl_loss.append(torch.mean(torch.cat(batch_kl_loss, dim=0)).item())
            epoch_spectral_loss.append(torch.mean(torch.cat(batch_spectral_loss, dim=0)).item())
            epoch_bn_loss.append(torch.mean(torch.cat(batch_bn_loss, dim=0)).item())

        global_step += 1

    # log epoch loss
    if rank == 0:
        run.log(
            {
                "train/loss": sum(epoch_loss) / len(epoch_loss),
                "train/recon_loss": sum(epoch_recon_loss) / len(epoch_recon_loss),
                "train/kl_loss": sum(epoch_kl_loss) / len(epoch_kl_loss),
                "train/spectral_loss": sum(epoch_spectral_loss) / len(epoch_spectral_loss),
                "train/bn_loss": sum(epoch_bn_loss) / len(epoch_bn_loss)
            },
            step=global_step
        )

    return global_step


def epoch_validation(dataloader: DataLoader, model: AutoEncoder, global_step: int,
                     rank: int, world_size: int, run: wandb.run = None):

    epoch_loss = []
    epoch_recon_loss = []
    epoch_kl_loss = []

    for step, x in enumerate(tqdm(dataloader)):

        x = x.to(f'cuda:{rank}')
        # b, c, h, w = x.shape
        # device = x.device

        logits, _, _, kl_all, _ = model(x)
        reconstructions = DiscMixLogistic(logits).log_prob(x)

        # reconstruction loss
        rec_loss = - torch.sum(reconstructions, dim=1)

        final_kl, _, _ = kl_balancer(kl_all, kl_balance=False)
        loss = rec_loss + final_kl

        # save all loss terms to rank 0
        if not rank == 0:
            dist.gather(tensor=loss.detach(), dst=0)
            dist.gather(tensor=rec_loss.detach(), dst=0)
            dist.gather(tensor=final_kl.detach(), dst=0)
        else:

            batch_loss = [torch.zeros_like(loss) for _ in range(world_size)]
            batch_recon_loss = [torch.zeros_like(rec_loss) for _ in range(world_size)]
            batch_kl_loss = [torch.zeros_like(final_kl) for _ in range(world_size)]

            dist.gather(gather_list=batch_loss, tensor=loss.detach())
            dist.gather(gather_list=batch_recon_loss, tensor=rec_loss.detach())
            dist.gather(gather_list=batch_kl_loss, tensor=final_kl.detach())

            # get full Batch mean losses for iteration.
            epoch_loss.append(torch.mean(torch.cat(batch_loss, dim=0)).item())
            epoch_recon_loss.append(torch.mean(torch.cat(batch_recon_loss, dim=0)).item())
            epoch_kl_loss.append(torch.mean(torch.cat(batch_kl_loss, dim=0)).item())

    # log epoch loss
    if rank == 0:
        run.log(
            {
                "validation/loss": sum(epoch_loss) / len(epoch_loss),
                "validation/recon_loss": sum(epoch_recon_loss) / len(epoch_recon_loss),
                "validation/kl_loss": sum(epoch_kl_loss) / len(epoch_kl_loss),
            },
            step=global_step
        )

def main(rank: int, world_size: int, args: argparse.Namespace, config: dict):

    # init wandb logger
    log_to_wandb = bool(args.logging)
    project_name = str(args.wandb_project)
    wandb_id = args.wandb_id

    if rank == 0:
        run = wandb.init(project=project_name, name=args.run_name, mode="offline" if not log_to_wandb else "online",
                         resume="must" if wandb_id is not None else None, id=wandb_id)
    else:
        run = None

    train_conf = config['training']

    setup(rank, world_size, train_conf)

    # Get data loaders.
    train_loader, val_loader, bpd_coefficient = prepare_data(rank, world_size, args.data_path, config)

    # create model and move it to GPU with id rank
    model = AutoEncoder(config['autoencoder'], config['resolution']).to(rank)

    if rank == 0:
        print(summary(model, torch.zeros((1,) + tuple(config['resolution']), device=f'cuda:{rank}'), show_input=False))

    ddp_model = DDP(model, device_ids=[rank])

    # find final learning rate
    base_learning_rate = float(train_conf['base_lr'])
    min_learning_rate = float(train_conf['min_lr'])
    weight_decay = float(train_conf['weight_decay'])
    eps = float(train_conf['eps'])

    final_learning_rate = base_learning_rate * math.sqrt(int(train_conf['cumulative_bs']) / 512)
    final_min_learning_rate = min_learning_rate * math.sqrt(int(train_conf['cumulative_bs']) / 512)

    if rank == 0:
        print(f'[INFO] final learning rate: {final_learning_rate}')
        print(f'[INFO] final min learning rate: {final_min_learning_rate}')

    # optimizer, scheduler, scaler
    optimizer = torch.optim.Adamax(ddp_model.parameters(), final_learning_rate, weight_decay=weight_decay, eps=eps)

    total_training_steps = len(train_loader) * train_conf['epochs']
    if train_conf['warmup_epochs'] is not None:
        warmup_steps = int(len(train_loader) * train_conf['warmup_epochs'])
        scheduler = LinearCosineScheduler(0, total_training_steps, final_learning_rate,
                                          final_min_learning_rate, warmup_steps)
    else:
        scheduler = CosineScheduler(0, total_training_steps, final_learning_rate, final_min_learning_rate)

    grad_scalar = GradScaler(2 ** 10)  # scale gradient for AMP

    # load checkpoint if resume
    if args.resume_from is not None:

        if rank == 0:
            print(f'[INFO] Loading checkpoint from: {args.resume_from}')

        checkpoint = torch.load(args.resume_from, map_location=f'cuda:{rank}')

        ddp_model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        grad_scalar.load_state_dict(checkpoint['grad_scalar'])

        global_step = checkpoint['global_step']
        init_epoch = checkpoint['epoch']

    else:
        global_step, init_epoch = 0, 0

    for epoch in range(init_epoch, train_conf['epochs']):

        if rank == 0:
            print(f'[INFO] Epoch {epoch+1}/{train_conf["epochs"]}')
            run.log({'epoch': epoch}, step=global_step)

        # Training
        ddp_model.train()
        # global_step = epoch_train(train_loader, ddp_model.module, optimizer, scheduler, grad_scalar,
        #                           train_conf["kl_anneal"], train_conf["spectral_regularization"],
        #                           total_training_steps, global_step, rank, world_size, run)

        # Validation
        dist.barrier()
        eval_freq = 1 if train_conf["epochs"] <= 50 else 20

        if epoch % eval_freq == 0 or epoch == (train_conf["epochs"] - 1):

            if rank == 0:
                print('[INFO] Validating')

            ddp_model.eval()
            with torch.no_grad():

                epoch_validation(val_loader, ddp_model.module, global_step, rank, world_size, run)

                if rank == 0:

                    num_samples = 8

                    # log reconstructions
                    x = next(iter(val_loader))[:num_samples].to(f"cuda:{rank}")
                    logits = ddp_model.module.autoencode(x)
                    output_imgs = DiscMixLogistic(logits, num_bits=8).mean()
                    output_imgs = make_grid(output_imgs).permute(1, 2, 0).cpu().numpy()
                    run.log({f"validation/reconstructions": output_imgs}, step=global_step)

                    # log samples
                    for t in [0.7, 0.8, 0.9, 1.0]:
                        logits = ddp_model.module.sample(num_samples, t)
                        output_imgs = DiscMixLogistic(logits, num_bits=8).sample()
                        output_imgs = make_grid(output_imgs).permute(1, 2, 0).cpu().numpy()
                        run.log({f"validation/samples tau={t:.2f}": output_imgs}, step=global_step)

            # Save checkpoint (after validation)
            if rank == 0:
                print(f'[INFO] Saving Checkpoint')

                ckpt_file = f"{args.checkpoint_base_path}/{args.run_name}/epoch={epoch}.pt"
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(), 'global_step': global_step,
                            'grad_scalar': grad_scalar.state_dict()}, ckpt_file)

    if rank == 0:
        wandb.finish()

    cleanup()


if __name__ == '__main__':

    w_size = torch.cuda.device_count()
    print(f'Using {w_size} gpus for training model')

    cudnn.benchmark = True

    arguments = parse_args()

    mp.spawn(main, args=(w_size, arguments, get_model_conf(arguments.conf_file),), nprocs=w_size)
