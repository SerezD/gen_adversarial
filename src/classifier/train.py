import argparse
import os
import warnings
import socket

import numpy as np

import torch
import torch.distributed as dist
from kornia.enhance import Normalize
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from pytorch_model_summary import summary
from kornia.augmentation import AugmentationSequential, RandomHorizontalFlip
from tqdm import tqdm

from data.loading_utils import ffcv_labels_loader
from src.classifier.model import ResNet


def parse_args():

    parser = argparse.ArgumentParser('Resnet50 training on Celeba Identities - 307 classes')

    parser.add_argument('--run_name', type=str, default='test')
    parser.add_argument('--data_path', type=str, required=True,
                        help='directory of ffcv datasets (.beton files)')

    parser.add_argument('--cumulative_bs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--checkpoint_base_path', type=str, default='./runs/',
                        help='directory where checkpoints are saved')

    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--resume_from', type=str, default=None,
                        help='if specified, resume training from this checkpoint')

    args = parser.parse_args()

    if WORLD_RANK == 0:

        # check data dir exists
        if (not os.path.exists(f'{args.data_path}/train.beton') or
                not os.path.exists(f'{args.data_path}/validation.beton')):
            raise FileNotFoundError(f'{args.data_path}/train.beton or {args.data_path}/validation.beton does not exist')

        # check checkpoint file exists
        if args.resume_from is not None and not os.path.exists(f'{args.resume_from}'):
            raise FileNotFoundError(f'could not find the specified checkpoint: {args.resume_from}')

        # create checkpoint out directory
        if not os.path.exists(f'{args.checkpoint_base_path}/{args.run_name}'):
            os.makedirs(f'{args.checkpoint_base_path}/{args.run_name}')

    return args

def setup(rank: int, world_size: int, seed: int):

    if 'MASTER_ADDR' not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        if rank == 0:
            warnings.warn("ENV VARIABLE 'MASTER_ADDR' not specified. Setting 'MASTER_ADDR'='localhost'")

    if 'MASTER_PORT' not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
        if rank == 0:
            warnings.warn("ENV VARIABLE 'MASTER_PORT' not specified. 'MASTER_PORT'='29500'")

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # ensures that weight initializations are all the same
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dist.barrier()


def cleanup():
    dist.destroy_process_group()


def prepare_data(rank: int, world_size: int, args: argparse.Namespace):

    image_size = 256
    batch_size = args.cumulative_bs // world_size
    seed = args.seed
    is_distributed = world_size > 1

    train_dataloader = ffcv_labels_loader(args.data_path, batch_size, image_size, seed, rank, is_distributed)
    val_dataloader = ffcv_labels_loader(args.data_path, batch_size, image_size, seed, rank, is_distributed,
                                        mode='validation')

    train_augmentations = AugmentationSequential(RandomHorizontalFlip(p=0.5),
                                                 Normalize(mean=0.5, std=0.5),
                                                 same_on_batch=False)
    val_augmentations = Normalize(mean=0.5, std=0.5)

    if WORLD_RANK == 0:
        line = f"[INFO] final batch size per device: {batch_size}\n"
        print(line)
        args.log.append(line)

    return train_dataloader, val_dataloader, train_augmentations, val_augmentations


def epoch_train(dataloader: DataLoader, augmentations: AugmentationSequential,
                model: ResNet, optimizer: torch.optim.Optimizer, args: argparse.Namespace, global_step: int):
    """
    :param dataloader: train dataloader.
    :param augmentations: augmentation module.
    :param model: model in training mode. Remember to pass ".module" with DDP.
    :param optimizer: optimizer object from torch.optim.Optimizer.
    :param args:
    :param global_step: for monitoring total number of steps.
    """

    epoch_losses = []

    for step, x in enumerate(tqdm(dataloader)):

        x, y = augmentations(x[0]), x[1]

        optimizer.zero_grad()

        # forward pass
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y.squeeze(1))

        loss.backward()
        optimizer.step()

        # to log at each step
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss = (loss / WORLD_SIZE).detach().item()

        epoch_losses.append(loss)

        global_step += 1

    # log epoch loss
    if WORLD_RANK == 0:

        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        line = f'[INFO] step: {global_step} - training loss: {epoch_loss:.4f}\n'
        print(line)
        args.log.append(line)

    return global_step


def epoch_validation(dataloader: DataLoader, model: ResNet, augmentations, args, global_step: int):

    all_preds = torch.empty((0, 1), device=f"cuda:{LOCAL_RANK}")
    all_targets = torch.empty((0, 1), device=f"cuda:{LOCAL_RANK}")

    for step, batch in enumerate(tqdm(dataloader)):

        x, y = augmentations(batch[0]), batch[1]

        logits = model(x)
        preds = torch.argmax(logits, dim=1, keepdim=True)

        all_preds = torch.cat((all_preds, preds))
        all_targets = torch.cat((all_targets, y))

    # compute accuracy
    n = all_preds.shape[0]
    correct_preds = (all_preds.eq(all_targets)).sum()

    dist.all_reduce(correct_preds, op=dist.ReduceOp.SUM)
    accuracy = correct_preds / (n * WORLD_SIZE)

    if WORLD_RANK == 0:
        line = f'[INFO] step: {global_step} - accuracy: {accuracy * 100:.2f}\n'
        print(line)
        args.log.append(line)


def main(args: argparse.Namespace):

    setup(WORLD_RANK, WORLD_SIZE, args.seed)

    args.log = []

    # Get data loaders.
    train_loader, val_loader, train_augmentations, val_augmentations = prepare_data(LOCAL_RANK, WORLD_SIZE, args)

    # create model and move it to GPU with id rank
    model = ResNet().to(LOCAL_RANK)

    # load checkpoint if resume
    if args.resume_from is not None:

        if WORLD_RANK == 0:
            line = f'[INFO] Loading checkpoint from: {args.resume_from}\n'
            print(line)
            args.log.append(line)

        checkpoint = torch.load(args.resume_from, map_location=f'cuda:{LOCAL_RANK}')
        model.load_state_dict(checkpoint['state_dict'])
        global_step = checkpoint['global_step']
        init_epoch = checkpoint['epoch']

        if WORLD_RANK == 0:
            line = f'[INFO] Start from Epoch: {init_epoch} - Step: {global_step}\n'
            print(line)
            args.log.append(line)

    else:
        global_step, init_epoch = 0, 0

        if WORLD_RANK == 0:
            line = summary(model, torch.zeros((1, 3, 256, 256), device=f'cuda:{LOCAL_RANK}'), show_input=False)
            print(line)
            args.log.append(line)

    # find final learning rate
    learning_rate = float(args.lr)

    if WORLD_RANK == 0:
        line = f'[INFO] final learning rate: {learning_rate}\n'
        print(line)
        args.log.append(line)

    # ddp model, optimizer, scheduler, scaler
    ddp_model = DDP(model, device_ids=[LOCAL_RANK])

    optimizer = torch.optim.AdamW(ddp_model.parameters(), learning_rate)

    for epoch in range(init_epoch, args.epochs):

        if WORLD_RANK == 0:
            line = f'[INFO] Epoch {epoch+1}/{args.epochs}\n'
            print(line)
            args.log.append(line)

        # Training
        ddp_model.train()
        global_step = epoch_train(train_loader, train_augmentations, ddp_model.module, optimizer, args, global_step)

        # Validation
        dist.barrier()
        eval_freq = 1 if args.epochs <= 50 else 5

        if epoch % eval_freq == 0 or epoch == (args.epochs - 1):

            if WORLD_RANK == 0:
                line = '[INFO] Validating\n'
                print(line)
                args.log.append(line)

            ddp_model.eval()
            with torch.no_grad():
                epoch_validation(val_loader, ddp_model.module, val_augmentations, args, global_step)

            # Save checkpoint (after validation)
            if WORLD_RANK == 0:
                line = '[INFO] Saving Checkpoint\n'
                print(line)
                args.log.append(line)

                ckpt_file = f"{args.checkpoint_base_path}/{args.run_name}/epoch={epoch:02d}.pt"
                torch.save({'epoch': epoch + 1, 'global_step': global_step,
                            'state_dict': ddp_model.module.state_dict()},
                           ckpt_file)

            dist.barrier()

    if WORLD_RANK == 0:
        line = '[INFO] Saving Checkpoint\n'
        print(line)
        args.log.append(line)

        ckpt_file = f"{args.checkpoint_base_path}/{args.run_name}/last.pt"
        torch.save({'epoch': args.epochs, 'global_step': global_step,
                    'state_dict': ddp_model.module.state_dict()},
                   ckpt_file)

        with open(f"{args.checkpoint_base_path}/{args.run_name}/log.txt", 'w') as f:
            f.writelines(args.log)

    cleanup()


if __name__ == '__main__':

    # on multinode cluster use mpi (enables to run only from master node)
    # on single_node local environment use torchrun
    # for debugging do not use anything (works only on 1 gpu)

    # torchrun --nproc_per_node=ngpus --nnodes=1 --node_rank=0 --master_addr='localhost'
    # --master_port=1234 main.py --> args

    # mpirun -np world_size -H ip_node_0:n_gpus,ip_node_1:n_gpus ... -x MASTER_ADDR=ip_master -x MASTER_PORT=1234
    # -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib
    # python main.py --args

    # Environment variables set by torch.distributed.launch or mpirun
    if 'LOCAL_RANK' in os.environ:
        # launched with torch distributed run
        LOCAL_RANK = int(os.environ['LOCAL_RANK'])
        WORLD_SIZE = int(os.environ['WORLD_SIZE'])
        WORLD_RANK = int(os.environ['RANK'])
    elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        # launched with ompi
        LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        WORLD_RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])
    else:
        # launched with standard python, debugging only
        print('[INFO] DEBUGGING MODE on single gpu!')
        LOCAL_RANK = 0
        WORLD_SIZE = 1
        WORLD_RANK = 0

    if LOCAL_RANK == 0:
        print(f'[INFO] STARTING on NODE: {socket.gethostname()}')

    if WORLD_RANK == 0:
        print(f'[INFO] Total number of processes: {WORLD_SIZE}')

    cudnn.benchmark = True

    arguments = parse_args()

    try:
        main(arguments)
    except KeyboardInterrupt as e:
        dist.destroy_process_group()
        raise e
