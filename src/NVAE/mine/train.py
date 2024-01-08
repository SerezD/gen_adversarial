import argparse
import math
import os
import numpy as np

import torch
from scheduling_utils.schedulers_cpp import LinearCosineScheduler, CosineScheduler
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader

from data.datasets import ImageDataset
from src.NVAE.mine.model import AutoEncoder


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

    args = parser.parse_args()

    # check data dir exists
    if not os.path.exists(f'{args.data_path}/train/') or not os.path.exists(f'{args.data_path}/validation/'):
        raise FileNotFoundError(f'{args.data_path}/train or {args.data_path}/validation does not exist')

    # check checkpoint file exists
    if args.resume_from is not None and not os.path.exists(f'{args.resume_from}/'):
        raise FileNotFoundError(f'could not find the specified checkpoint: {args.resume_from}')

    # create checkpoint out directory
    if not os.path.exists(f'{args.checkpoint_base_path}/{args.run_name}'):
        os.makedirs(f'{args.checkpoint_base_path}/{args}')

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
        print([f"[INFO] final batch size per device: {batch_size}"])
        print([f"[INFO] Bit Per Dimension (bpd) coefficient: {bpd_coefficient}"])

    return train_dataloader, val_dataloader, bpd_coefficient


def train():
    pass


def validate():
    pass


def main(rank: int, world_size: int, args: argparse.Namespace, config: dict):

    train_conf = config['train']

    setup(rank, world_size, train_conf)

    # Get data loaders.
    train_loader, val_loader, bpd_coefficient = prepare_data(rank, world_size, args.data_path, config)

    # create model and move it to GPU with id rank
    model = AutoEncoder(config['autoencoder'], config['resolution']).to(rank)
    ddp_model = DDP(model.to(rank), device_ids=[rank])

    # find final learning rate
    base_learning_rate = train_conf['base_lr']
    min_learning_rate = train_conf['min_lr']
    weight_decay = train_conf['weight_decay']
    eps = train_conf['eps']

    final_learning_rate = base_learning_rate * math.sqrt(train_conf['cumulative_bs'] / 512)
    final_min_learning_rate = min_learning_rate * math.sqrt(train_conf['cumulative_bs'] / 512)

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

        # Training (TODO).
        ddp_model.train()
        train()

        # Validation
        eval_freq = 1 if args.epochs <= 50 else 20

        if epoch % eval_freq == 0 or epoch == (train_conf["epochs"] - 1):

            ddp_model.eval()
            with torch.no_grad():
                num_samples = 8
                n = int(np.floor(np.sqrt(num_samples)))
                for t in [0.7, 0.8, 0.9, 1.0]:
                    logits = ddp_model.sample(num_samples, t)
                    output = ddp_model.decoder_output(logits)
                    output_img = output.sample(t)
                    # output_tiled = utils.tile_image(output_img, n)
                    # writer.add_image('generated_%0.1f' % t, output_tiled, global_step)

                validate() # TODO

            # Save checkpoint (after validation)
            if rank == 0:
                print(f'[INFO] Saving Checkpoint')

                ckpt_file = f"{args.checkpoint_base_path}/{args.run_name}/epoch={epoch}.pt"
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(), 'global_step': global_step,
                            'grad_scalar': grad_scalar.state_dict()}, ckpt_file)

    cleanup()


if __name__ == '__main__':

    w_size = torch.cuda.device_count()
    print(f'Using {w_size} gpus for training model')

    cudnn.benchmark = True

    arguments = parse_args()

    mp.spawn(main, args=(w_size, arguments, get_model_conf(arguments.conf_file),))
