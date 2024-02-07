import foolbox as fb
import eagerpy as ep
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from data.datasets import ImageLabelDataset
from src.final_experiments.common_utils import load_hub_CNN, load_NVAE


def main(data_path: str, samples_to_test: int, batch_size: int, cnn_path: str, cnn_type: str, nvae_path: str,
         device: str):

    nvae = load_NVAE(nvae_path, device)

    attacked_model = fb.PyTorchModel(load_hub_CNN(cnn_path, cnn_type, device),
                                     bounds=(0, 1),
                                     preprocessing=dict(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225],
                                         axis=-3),
                                     device=device)

    # load whole validation set as single batch
    # foolbox wants ep tensors for working faster
    dataloader = DataLoader(ImageLabelDataset(folder=data_path, image_size=32),
                            batch_size=samples_to_test, shuffle=True)
    images, labels = next(iter(dataloader))
    images, labels = ep.astensors(*(images.to(device), labels.to(device)))

    # TEST: Clean Accuracy according to official repo is 93.53% for Resnet and 94.16% for VGG
    clean_acc = fb.accuracy(attacked_model, images, labels)
    print(f"clean accuracy:  {clean_acc * 100:.2f} %")

    # attacks to test
    attacks = [
        fb.attacks.L2PGD(),
        fb.attacks.LinfPGD(),
        fb.attacks.L2FastGradientAttack(),
        fb.attacks.LinfFastGradientAttack(),
        fb.attacks.L2DeepFoolAttack(),
        fb.attacks.LinfDeepFoolAttack(),
        fb.attacks.DDNAttack(),
        fb.attacks.BoundaryAttack(),
        fb.attacks.HopSkipJumpAttack()
    ]

    epsilons = [(0.5,), (8/255,), (0.5,), (8/255,), (0.5,), (8/255,), (0.5,), (0.5,), (0.5,)]


    for (attack, eps) in zip(attacks, epsilons):

        print('#' * 20)
        print(f'Attack: {attack} - epsilon: {eps}')

        _, adv, success = attack(attacked_model, images, labels, epsilons=eps)

        print(f'Success rate: {success.sum().item() / samples_to_test}')
        # acc = fb.accuracy(attacked_model, adv[0], labels)
        # print(f"robust accuracy:  {acc * 100:.2f} %")  # This is 1 - success rate

        with torch.no_grad():

            # check where chunks have changed the most
            valid_cln = images[success.squeeze(0)].raw
            valid_adv = adv[0][success.squeeze(0)].raw
            b = valid_cln.shape[0]

            # plot and save one single batch of images (clean vs adv)
            n = min(8, samples_to_test)
            name = str(attack).split('(')[0] + f'_eps={eps[0]:.3f}'
            display = make_grid(
                torch.cat([images.raw[:n], adv[0].raw[:n]]),
                nrow=n).permute(1, 2, 0).cpu().numpy()
            plt.imshow(display)
            plt.title(name)
            plt.axis(False)
            plt.savefig(f'./attacks_figures/{name}.png')
            plt.close()

            chunks_dists = torch.empty((3, 0), device=device)

            for i in range(0, b, batch_size):

                # drop last = True
                if i + batch_size > b:
                    continue

                x1 = valid_cln[i:i + batch_size]
                x2 = valid_adv[i:i + batch_size]

                chunks_cln, std_cln = nvae.encode(x1, deterministic=True, with_std=True)
                chunks_adv = nvae.encode(x2, deterministic=True)

                batched_dists = torch.empty((0, batch_size), device=device)

                for i, (cc, ca, sc) in enumerate(zip(chunks_cln, chunks_adv, std_cln)):

                    # get chunks and normalize them
                    ca_norm = (ca.view(batch_size, -1) - cc.view(batch_size, -1)) / sc.view(batch_size, -1)

                    # compute L2 norm (like l2 dist with first term 0)
                    l2_dist = torch.norm(ca_norm, p=2, dim=1)
                    normalized_l2_dist = l2_dist / ca_norm.shape[1]

                    batched_dists = torch.cat((batched_dists, normalized_l2_dist.unsqueeze(0)), dim=0)

                    # TODO ASK WHAT WE SHOULD DO WITH KL DIVERGENCE
                    # P = cc.view(b, -1)
                    # Q = ca.view(b, -1) + 1e-20
                    # print(f'\tKL divergence: {(P * (P / Q).log()).sum().item()}')

                chunks_dists = torch.cat((chunks_dists, batched_dists), dim=1)

            for i, c in enumerate(chunks_dists):

                print(f'chunk {i}:')
                print(f'\t mean: {c.mean().item()}')
                print(f'\t std: {c.std().item()}')


if __name__ == '__main__':

    # TODO parse some args

    main(
        data_path='/media/dserez/datasets/cifar10/validation/',
        samples_to_test=16, # 10000,
        batch_size=8,
        cnn_path='/media/dserez/runs/adversarial/CNNs/',
        cnn_type='vgg16',  # 'resnet32' 'vgg16'
        nvae_path='/media/dserez/runs/NVAE/cifar10/ours/replica/large_latents.pt',
        device='cuda:0'
    )