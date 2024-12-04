import argparse
import kornia
import os
import numpy as np
import socket
import torch
import warnings

from kornia.augmentation import AugmentationSequential, RandomHorizontalFlip, RandomResizedCrop, RandomGrayscale, \
    RandomContrast, RandomEqualize, RandomBrightness
from kornia.enhance import Normalize
from torch import distributed as dist
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel as ddp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data.datasets import ImageLabelDataset
from src.classifier.model import ResNet, Vgg, ResNext
from src.defenses.competitors.trades.modules import trades_loss


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser('Classifier Fine Tuning for Adversarial Robustness with Trades')

    parser.add_argument('--run_name', type=str)
    parser.add_argument('--data_path', type=str, required=True,
                        help='directory with train, validation subdirectories')
    parser.add_argument('--model_type', type=str, choices=['resnext', 'resnet', 'vgg'])
    parser.add_argument('--n_classes', type=int)
    parser.add_argument('--beta', type=float, help='In range 1.0, 10.0, '
                                                   'the higher means robustness is considered more than classification.')

    parser.add_argument('--resume_from', type=str,
                        help='pretrained model for fine tuning')

    parser.add_argument('--cumulative_bs', type=int)
    parser.add_argument('--image_size', type=int)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--checkpoint_base_path', type=str,
                        help='directory where new checkpoints are saved')

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)

    args = parser.parse_args()

    if WORLD_RANK == 0:

        # check data dir exists
        if (not os.path.exists(f'{args.data_path}/train') or
                not os.path.exists(f'{args.data_path}/validation')):
            raise FileNotFoundError(f'{args.data_path}/train or {args.data_path}/validation does not exist')

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


def prepare_data(world_size: int, args: argparse.Namespace) -> \
        [DataLoader, DataLoader, AugmentationSequential, Normalize]:

    image_size = args.image_size
    batch_size = args.cumulative_bs // world_size
    is_distributed = world_size > 1

    t_dataset = ImageLabelDataset(f'{args.data_path}/train', image_size)
    v_dataset = ImageLabelDataset(f'{args.data_path}/validation', image_size)

    if is_distributed:
        train_dataloader = DataLoader(t_dataset, batch_size=batch_size, shuffle=False,
                                      sampler=DistributedSampler(t_dataset))
        val_dataloader = DataLoader(v_dataset, batch_size=batch_size, shuffle=False,
                                    sampler=DistributedSampler(v_dataset))
    else:
        train_dataloader = DataLoader(t_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(v_dataset, batch_size=batch_size, shuffle=False)

    train_augmentations = AugmentationSequential(RandomHorizontalFlip(p=0.5),
                                                 RandomResizedCrop(size=(image_size, image_size),
                                                                   scale=(0.75, 1.0)),
                                                 same_on_batch=False)
    normalization_function = Normalize(mean=0.5, std=0.5)

    if WORLD_RANK == 0:
        line = f"[INFO] final batch size per device: {batch_size}\n"
        print(line)
        args.log.append(line)

    return train_dataloader, val_dataloader, train_augmentations, normalization_function


def epoch_train(dataloader: DataLoader, augmentations: AugmentationSequential, normalization_function: Normalize,
                model: ResNet, optimizer: torch.optim.Optimizer, args: argparse.Namespace, global_step: int) -> int:

    epoch_losses = []

    for step, x in enumerate(tqdm(dataloader)):

        x, y = augmentations(x[0].to(LOCAL_RANK)), x[1].to(LOCAL_RANK)

        optimizer.zero_grad()

        # forward pass
        loss = trades_loss(model, x, y, optimizer,
                           perturb_steps=16,
                           distance='l_2',
                           epsilon=2.0 if args.model_type == 'vgg' else 4.0,
                           beta=args.beta,
                           step_size=0.001,
                           normalization_function=normalization_function)  # ADDED

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


def epoch_validation(dataloader: DataLoader, model: ResNet, normalization_function, args, global_step: int):

    all_preds = torch.empty((0, 1), device=f"cuda:{LOCAL_RANK}")
    all_targets = torch.empty((0, 1), device=f"cuda:{LOCAL_RANK}")

    for step, batch in enumerate(tqdm(dataloader)):

        x, y = normalization_function(batch[0].to(LOCAL_RANK)), batch[1].to(LOCAL_RANK)

        logits = model(x)
        preds = torch.argmax(logits, dim=1, keepdim=True)

        all_preds = torch.cat((all_preds, preds))
        all_targets = torch.cat((all_targets, y.unsqueeze(1)))

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
    train_loader, val_loader, train_augmentations, normalization_function = prepare_data(WORLD_SIZE, args)

    # create model and move it to GPU with id rank
    if args.model_type == 'resnext':
        model = ResNext(n_classes=args.n_classes, get_weights=False).to(LOCAL_RANK)
    elif args.model_type == 'resnet':
        model = ResNet(n_classes=args.n_classes, get_weights=False).to(LOCAL_RANK)
    elif args.model_type == 'vgg':
        model = Vgg(n_classes=args.n_classes, get_weights=False).to(LOCAL_RANK)
    else:
        raise NotImplementedError

    # load checkpoint
    if WORLD_RANK == 0:
        line = f'[INFO] Loading checkpoint from: {args.resume_from}\n'
        print(line)
        args.log.append(line)

    checkpoint = torch.load(args.resume_from, map_location=f'cuda:{LOCAL_RANK}')
    model.load_state_dict(checkpoint['state_dict'])
    global_step = 0
    init_epoch = 0

    # find final learning rate
    learning_rate = float(args.lr)

    if WORLD_RANK == 0:
        line = f'[INFO] final learning rate: {learning_rate}\n'
        print(line)
        args.log.append(line)

    # ddp model, optimizer, scheduler, scaler
    ddp_model = ddp(model, device_ids=[LOCAL_RANK])

    optimizer = torch.optim.SGD(ddp_model.parameters(), learning_rate, momentum=0.9)

    ddp_model.eval()
    with torch.no_grad():
        epoch_validation(val_loader, ddp_model.module, normalization_function, args, global_step)

    for epoch in range(init_epoch, args.epochs):

        if WORLD_RANK == 0:
            line = f'[INFO] Epoch {epoch+1}/{args.epochs}\n'
            print(line)
            args.log.append(line)

        # Training
        ddp_model.train()

        if WORLD_SIZE > 1:
            train_loader.sampler.set_epoch(epoch)

        global_step = epoch_train(train_loader, train_augmentations, normalization_function,
                                  ddp_model.module, optimizer, args, global_step)

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
                epoch_validation(val_loader, ddp_model.module, normalization_function, args, global_step)

            # Save checkpoint (after validation)
            if WORLD_RANK == 0 and epoch % (eval_freq * 2) == 0:
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
