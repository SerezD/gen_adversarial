"""
take different autoattack attacks.
encode adversaries with nvae and check distance from origin w.r.t clean
"""
import math

import torch
import os
import time

from kornia.enhance import Normalize
from matplotlib import pyplot as plt
from torchmetrics import Accuracy
from tqdm import tqdm

from autoattack.autopgd_base import APGDAttack, APGDAttack_targeted
from autoattack.fab_pt import FABAttack_PT
from autoattack.square import SquareAttack
from robustbench import load_cifar10
from torchvision.utils import make_grid

from src.NVAE.original.model import AutoEncoder
from src.NVAE.original.utils import get_arch_cells

@torch.no_grad()
def main():

    # load nvae pretrained cifar10
    checkpoint = torch.load(CKPT_NVAE, map_location='cpu')

    # get and update args
    args = checkpoint['args']
    args.num_mixture_dec = 10
    args.batch_size = BATCH_SIZE

    # init model and load
    arch_instance = get_arch_cells(args.arch_instance)
    nvae = AutoEncoder(args, None, arch_instance)
    nvae.load_state_dict(checkpoint['state_dict'], strict=False)
    nvae = nvae.to('cuda:0').eval()

    # load cifar10
    x_test, y_test = load_cifar10(data_dir=ADV_BASE_PATH)
    data_n = x_test.shape[0]
    # data_n = 100
    x_test = x_test[:data_n]
    y_test = y_test[:data_n]

    # cnn to test
    os.environ["TORCH_HOME"] = TORCH_HOME
    resnet32 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True)
    resnet32 = resnet32.to('cuda:1').eval()
    vgg16 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
    vgg16 = vgg16.to('cuda:1').eval()

    accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1).to('cuda:0')
    cnn_preprocess = Normalize(mean=torch.tensor([0.507, 0.4865, 0.4409], device='cuda:1'),
                               std=torch.tensor([0.2673, 0.2564, 0.2761], device='cuda:1'))

    for attacked_model, model_name in zip([resnet32, vgg16], ['resnet32', 'vgg16']):

        for metric, bound in zip(['Linf', 'L2'], [8/255, 0.5]):

            predict_pass = lambda x: attacked_model(cnn_preprocess(x))
            for attack, attack_name in zip(
                    [APGDAttack(predict_pass, norm=metric, eps=bound),
                     FABAttack_PT(predict_pass, norm=metric, eps=bound),
                     SquareAttack(predict_pass, norm=metric, eps=bound)],
                    ['APGD', 'FAB', 'SQUARE']):

                print('#' * 30)
                print(f'Attacked Model: {model_name} - Attack: {attack_name} with {metric}={bound:.4f}')

                adv_test = torch.empty((0, 3, 32, 32), device='cuda:1')
                def_test = torch.empty((0, sum(nvae.groups_per_scale), 3, 32, 32), device='cuda:1')
                mu_cln_mu_adv_distances = torch.empty((0, sum(nvae.groups_per_scale), 1), device='cuda:1')
                ck_cln_mu_cln_distances = torch.empty((0, sum(nvae.groups_per_scale), 1), device='cuda:1')
                ck_adv_mu_cln_distances = torch.empty((0, sum(nvae.groups_per_scale), 1), device='cuda:1')
                ck_cln_ck_adv_distances = torch.empty((0, sum(nvae.groups_per_scale), 1), device='cuda:1')

                time.sleep(0.5)
                for b in tqdm(range(0, data_n, BATCH_SIZE)):

                    x = x_test[b:b + BATCH_SIZE]
                    y = y_test[b: b + BATCH_SIZE]

                    # compute x_adv
                    x_adv = attack.perturb(x.to('cuda:1'), y.to('cuda:1'))
                    adv_test = torch.cat([adv_test, x_adv.to('cuda:1')], dim=0)

                    # encode and get chunks, mu
                    clean_chunks, clean_mu_chunks = nvae.encode(x.to('cuda:0'))
                    adv_chunks, adv_mu_chunks = nvae.encode(x_adv.to('cuda:0'))

                    # chunks for defense
                    def_chunks = [[] for _ in range(sum(nvae.groups_per_scale))]

                    # compute L2 distances
                    mu_cln_mu_adv_batch = torch.empty((BATCH_SIZE, 0, 1), device='cuda:1')
                    ck_cln_mu_cln_batch = torch.empty((BATCH_SIZE, 0, 1), device='cuda:1')
                    ck_adv_mu_cln_batch = torch.empty((BATCH_SIZE, 0, 1), device='cuda:1')
                    ck_cln_ck_adv_batch = torch.empty((BATCH_SIZE, 0, 1), device='cuda:1')

                    for i, (ck_cln, ck_adv, mu_cln, mu_adv) in enumerate(
                            zip(clean_chunks, adv_chunks, clean_mu_chunks, adv_mu_chunks)):

                        mu_cln_mu_adv_batch = torch.cat(
                            [mu_cln_mu_adv_batch,
                             torch.cdist(mu_cln.view(BATCH_SIZE, -1),
                                         mu_adv.view(BATCH_SIZE, -1), p=2).diag().view(-1, 1, 1).to('cuda:1')], dim=1)

                        ck_cln_mu_cln_batch = torch.cat(
                            [ck_cln_mu_cln_batch,
                             torch.cdist(ck_cln.view(BATCH_SIZE, -1),
                                         mu_cln.view(BATCH_SIZE, -1), p=2).diag().view(-1, 1, 1).to('cuda:1')], dim=1)

                        ck_adv_mu_cln_batch = torch.cat(
                            [ck_adv_mu_cln_batch,
                             torch.cdist(ck_adv.view(BATCH_SIZE, -1),
                                         mu_cln.view(BATCH_SIZE, -1), p=2).diag().view(-1, 1, 1).to('cuda:1')], dim=1)

                        ck_cln_ck_adv_batch = torch.cat(
                            [ck_cln_ck_adv_batch,
                             torch.cdist(ck_cln.view(BATCH_SIZE, -1),
                                         ck_adv.view(BATCH_SIZE, -1), p=2).diag().view(-1, 1, 1).to('cuda:1')], dim=1)

                        # compute all def_chunks
                        r = int(math.sqrt(ck_adv.shape[1] // nvae.num_latent_per_group))
                        for j, d_ck in enumerate(def_chunks):
                            if i == j:
                                d_ck.append(mu_cln)
                            else:
                                d_ck.append(ck_adv.view(BATCH_SIZE, nvae.num_latent_per_group, r, r))

                    # add computations for batch
                    mu_cln_mu_adv_distances = torch.cat([mu_cln_mu_adv_distances, mu_cln_mu_adv_batch], dim=0)
                    ck_cln_mu_cln_distances = torch.cat([ck_cln_mu_cln_distances, ck_cln_mu_cln_batch], dim=0)
                    ck_adv_mu_cln_distances = torch.cat([ck_adv_mu_cln_distances, ck_adv_mu_cln_batch], dim=0)
                    ck_cln_ck_adv_distances = torch.cat([ck_cln_ck_adv_distances, ck_cln_ck_adv_batch], dim=0)

                    # defense
                    x_def = torch.empty((BATCH_SIZE, 0, 3, 32, 32), device='cuda:0')
                    for def_ck in def_chunks:
                        x_def = torch.cat([x_def,
                                           nvae.decoder_output(nvae.decode(def_ck)).sample().unsqueeze(1)], dim=1)
                    def_test = torch.cat([def_test, x_def.to('cuda:1')], dim=0)

                # compute mean and std per chunk!
                mu_cln_mu_adv_means = torch.mean(mu_cln_mu_adv_distances, dim=0)
                mu_cln_mu_adv_stds = torch.std(mu_cln_mu_adv_distances, dim=0)
                ck_cln_mu_cln_means = torch.mean(ck_cln_mu_cln_distances, dim=0)
                ck_cln_mu_cln_stds = torch.std(ck_cln_mu_cln_distances, dim=0)
                ck_adv_mu_cln_means = torch.mean(ck_adv_mu_cln_distances, dim=0)
                ck_adv_mu_cln_stds = torch.std(ck_adv_mu_cln_distances, dim=0)
                ck_cln_ck_adv_means = torch.mean(ck_cln_ck_adv_distances, dim=0)
                ck_cln_ck_adv_stds = torch.std(ck_cln_ck_adv_distances, dim=0)

                # print TABLE
                print('\\begin{table}[]')
                print('\\begin{tabular}{l|ll|ll|ll|ll}')
                print('\multicolumn{1}{c|}{\multirow{2}{*}{chunks}} & \multicolumn{2}{l|}{dist(mu clean, mu adv)} & \multicolumn{2}{l|}{dist(ck clean, mu clean)} & \multicolumn{2}{l|}{dist(ck adv, mu clean)} & \multicolumn{2}{l}{dist(ck clean, ck adv)} \\\\')
                print('\multicolumn{1}{c|}{} & mu & sigma & mu & sigma & mu & sigma & mu & sigma \\\\ \hline')

                for i in range(sum(nvae.groups_per_scale)):

                    print(f'{i} &')
                    print(f'{mu_cln_mu_adv_means[i].item():.2f} & {mu_cln_mu_adv_stds[i].item():.2f} &')
                    print(f'{ck_cln_mu_cln_means[i].item():.2f} & {ck_cln_mu_cln_stds[i].item():.2f} &')
                    print(f'{ck_adv_mu_cln_means[i].item():.2f} & {ck_adv_mu_cln_stds[i].item():.2f} &')
                    print(f'{ck_cln_ck_adv_means[i].item():.2f} & {ck_cln_ck_adv_stds[i].item():.2f} \\\\')

                    if (i + 1) < sum(nvae.groups_per_scale) and (i + 1) % 2 == 0:
                        print('\\hline')

                print('\\bottomrule\n\end{tabular}\n\end{table}')

                # print ACCURACIES
                clean_preds = torch.argmax(attacked_model(cnn_preprocess(x_test.to('cuda:1'))), dim=1)
                adv_preds = torch.argmax(attacked_model(cnn_preprocess(adv_test.to('cuda:1'))), dim=1)

                print(f"clean acc: {accuracy(clean_preds.to('cuda:0'), y_test.to('cuda:0')):.3f} - "
                      f"atk acc: {accuracy(adv_preds.to('cuda:0'), y_test.to('cuda:0')):.3f} - ")

                for i in range(sum(nvae.groups_per_scale)):
                    def_preds = torch.argmax(attacked_model(cnn_preprocess(def_test[:, i].to('cuda:1'))), dim=1)
                    print(f"def {i} acc {accuracy(def_preds.to('cuda:0'), y_test.to('cuda:0')):.3f} - ")

                # plot an example
                if metric == 'Linf':
                    test_res = torch.mean(torch.cdist(x_test.to('cuda:1').view(data_n, -1), adv_test.to('cuda:1').view(data_n, -1), p=float('inf')).diag())
                else:
                    test_res = torch.mean(torch.cdist(x_test.to('cuda:1').view(data_n, -1), adv_test.to('cuda:1').view(data_n, -1), p=2).diag())

                tensors = [x_test[:8].to('cuda:1'), adv_test[:8].to('cuda:1')]
                for i in range(sum(nvae.groups_per_scale)):
                    tensors.append(def_test[:8, i].to('cuda:1'))

                plt.imshow(make_grid(torch.cat(tensors, dim=0)).permute(1, 2, 0).cpu().numpy())
                plt.axis(False)
                plt.suptitle(f"Attacked Model: {model_name} - Attack: {attack_name} with {metric}={bound:.4f}")
                plt.title(f"Metric: {metric} - Bound: {bound:.3f} - Measured: {test_res:.3f}")
                plt.savefig(f"{OUT_FILE}/{model_name}_{attack_name}_{metric}:{bound}.png")
                # plt.show()


if __name__ == '__main__':

    # CKPT_NVAE = '/media/dserez/runs/NVAE/cifar10/best/ours_weighted.pt'
    # ADV_BASE_PATH = '/media/dserez/code/adversarial/'
    # TORCH_HOME = '/media/dserez/runs/classification/cifar10/'
    # OUT_FILE = f'{ADV_BASE_PATH}images/'
    #
    # BATCH_SIZE = 100

    CKPT_NVAE = '/home/dserez/gen_adversarial/loading/ours_weighted.pt'
    ADV_BASE_PATH = '/home/dserez/gen_adversarial/loading/'
    TORCH_HOME = '/home/dserez/gen_adversarial/loading/'
    OUT_FILE = f'{ADV_BASE_PATH}defense_images/'

    BATCH_SIZE = 200

    main()