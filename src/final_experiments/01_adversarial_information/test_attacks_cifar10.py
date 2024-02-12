import foolbox as fb
import eagerpy as ep
import matplotlib.pyplot as plt
import torch
from einops import rearrange
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
        # fb.attacks.BoundaryAttack(),
        # fb.attacks.HopSkipJumpAttack()
    ]

    epsilons_linf = [(1/255,), (2/255,), (4/255,), (8/255,)]
    epsilons_l2 = [(0.1,), (0.2,), (0.3,), (0.5,)]

    for (e_inf, e_2) in zip(epsilons_linf, epsilons_l2):

        for attack in attacks:

            if (isinstance(attack, fb.attacks.L2PGD) or isinstance(attack, fb.attacks.L2FastGradientAttack) or
                isinstance(attack, fb.attacks.L2DeepFoolAttack) or isinstance(attack, fb.attacks.DDNAttack)):
                eps = e_2
            elif (isinstance(attack, fb.attacks.LinfPGD) or isinstance(attack, fb.attacks.LinfFastGradientAttack) or
                  isinstance(attack, fb.attacks.LinfDeepFoolAttack)):
                eps = e_inf
            else:
                eps = 1.

            print('#' * 20)
            print(f'Attack: {attack} - epsilon: {eps}')

            if isinstance(attack, fb.attacks.BoundaryAttack) or isinstance(attack, fb.attacks.HopSkipJumpAttack):
                with torch.no_grad():
                    _, adv, success = attack(attacked_model, images, labels, epsilons=eps)
            else:
                _, adv, success = attack(attacked_model, images, labels, epsilons=eps)

            print(f'Success rate: {success.sum().item() / samples_to_test}')
            # acc = fb.accuracy(attacked_model, adv[0], labels)
            # print(f"robust accuracy:  {acc * 100:.2f} %")  # This is 1 - success rate

            with torch.no_grad():
                # check where chunks have changed the most
                valid_cln = images.raw
                valid_adv = adv[0].raw
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

                # chunks_dists = torch.empty((3, 0), device=device)
                p_clean = torch.empty((3, 0, 64), device=device)
                q_adver = torch.empty((3, 0, 64), device=device)

                for i in range(0, b, batch_size):

                    # drop last = True
                    if i + batch_size > b:
                        continue

                    x1 = valid_cln[i:i + batch_size]
                    x2 = valid_adv[i:i + batch_size]

                    chunks_cln, std_cln = nvae.encode(x1, deterministic=True, with_std=True)
                    chunks_adv = nvae.encode(x2, deterministic=True)

                    # batched_dists = torch.empty((0, batch_size), device=device)
                    batched_q = torch.empty((0, batch_size, 64), device=device)
                    batched_p = torch.empty((0, batch_size, 64), device=device)

                    for i, (cc, ca, sc) in enumerate(zip(chunks_cln, chunks_adv, std_cln)):

                        # compute the normalized adversarial chunk (clean is 0)
                        ca = rearrange(ca, 'b n h w -> b n (h w)').mean(dim=2)
                        cc = rearrange(cc, 'b n h w -> b n (h w)').mean(dim=2)
                        sc = rearrange(sc, 'b n h w -> b n (h w)').mean(dim=2)

                        # TODO THIS IS METHOD ONE (NORMALIZE BY STD)
                        # ca_norm = (ca - cc) / sc
                        #
                        # # compute L2 norm (like l2 dist with first term 0)
                        # l2_dist = torch.norm(ca_norm, p=2, dim=1)
                        # # normalized_l2_dist = l2_dist / ca_norm.shape[1]

                        # TODO THIS IS METHOD TWO (TAKE THE PERC DIFFERENCE)
                        # ca_norm = ca / cc
                        # l2_dist = torch.cdist(ca_norm, torch.ones_like(cc), p=2).diag()
                        #
                        # batched_dists = torch.cat((batched_dists, l2_dist.unsqueeze(0)), dim=0)

                        # TODO KL DIVERGENCE
                        batched_p = torch.cat((batched_p, cc.unsqueeze(0)), dim=0)
                        batched_q = torch.cat((batched_q, ca.unsqueeze(0)), dim=0)

                    # chunks_dists = torch.cat((chunks_dists, batched_dists), dim=1)
                    p_clean = torch.cat((p_clean, batched_p), dim=1)
                    q_adver = torch.cat((q_adver, batched_q), dim=1)

                # for i, c in enumerate(chunks_dists):
                #
                #     print(f'chunk {i}:')
                #     print(f'\t mean: {c.mean().item()}')
                #     print(f'\t std: {c.std().item()}')

                for i, (c, a) in enumerate(zip(p_clean, q_adver)):

                    # Create distributions from the data
                    P_distribution = torch.distributions.MultivariateNormal(c.mean(dim=0), covariance_matrix=torch.diag(c.var(dim=0)))
                    Q_distribution = torch.distributions.MultivariateNormal(a.mean(dim=0), covariance_matrix=torch.diag(a.var(dim=0)))

                    # Compute the KL divergence
                    kl_divergence = torch.distributions.kl_divergence(P_distribution, Q_distribution)

                    print(f'chunk {i}:')
                    print(f'\t KL: {kl_divergence.item()}')



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