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

    accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1).to('cuda:3')
    cnn_preprocess = Normalize(mean=torch.tensor([0.507, 0.4865, 0.4409], device='cuda:1'),
                               std=torch.tensor([0.2673, 0.2564, 0.2761], device='cuda:1'))

    for attacked_model, model_name in zip([resnet32, vgg16], ['resnet32', vgg16]):

        for metric, bound in zip(['Linf', 'L2'], [8/255, 0.5]):

            predict_pass = lambda x: attacked_model(cnn_preprocess(x))
            for attack, attack_name in zip(
                    [APGDAttack(predict_pass, norm=metric, eps=bound),
                     FABAttack_PT(predict_pass, norm=metric, eps=bound),
                     SquareAttack(predict_pass, norm=metric, eps=bound)],
                    ['APGD', 'FAB', 'SQUARE']):

                print('#' * 30)
                print(f'Attacked Model: {model_name} - Attack: {attack_name} with {metric}={bound:.4f}')

                adv_test = torch.empty((0, 3, 32, 32), device='cuda:2')
                def_test = torch.empty((0, 3, 32, 32), device='cuda:2')
                origin_distances = torch.empty((0, sum(nvae.groups_per_scale), 1), device='cuda:2')

                time.sleep(1)
                for b in tqdm(range(0, data_n, BATCH_SIZE)):

                    x = x_test[b:b + BATCH_SIZE]
                    y = y_test[b: b + BATCH_SIZE]
                    x_adv = attack.perturb(x.to('cuda:1'), y.to('cuda:1'))
                    adv_test = torch.cat([adv_test, x_adv.to('cuda:2')], dim=0)

                    # get clean and adv chunks
                    with torch.no_grad():
                        clean_chunks = nvae.encode(x.to('cuda:0'))
                        adv_chunks = nvae.encode(x_adv.to('cuda:0'))
                        def_chunks = []

                        # compute L2 distances
                        dist_vector = torch.empty((BATCH_SIZE, 0, 1), device='cuda:2')

                        for i, (c_cln, c_adv) in enumerate(zip(clean_chunks, adv_chunks)):
                            origin_dist_clean = torch.cdist(c_cln, torch.zeros_like(c_cln, device='cuda:0'), p=2).diag().view(-1, 1, 1)
                            origin_dist_adver = torch.cdist(c_adv, torch.zeros_like(c_adv, device='cuda:0'), p=2).diag().view(-1, 1, 1)

                            dist = origin_dist_adver - origin_dist_clean
                            dist_vector = torch.cat([dist_vector, dist.to('cuda:2')], dim=1)

                            # append def_chunks or set 0
                            r = int(math.sqrt(c_adv.shape[1] // nvae.num_latent_per_group))
                            def_chunk = c_adv.view(BATCH_SIZE, nvae.num_latent_per_group, r, r)
                            if i < 5:
                                def_chunks.append(def_chunk)
                            else:
                                def_chunks.append(torch.zeros_like(def_chunk, device=def_chunk.device))

                        origin_distances = torch.cat([origin_distances, dist_vector], dim=0)

                        # defense
                        x_def = nvae.decoder_output(nvae.decode(def_chunks)).sample()
                        def_test = torch.cat([def_test, x_def.to('cuda:2')], dim=0)
                        del x; del y; del x_adv; del x_def; del def_chunks

                # compute mean and std per chunk!
                means = torch.mean(origin_distances, dim=0)
                stds = torch.std(origin_distances, dim=0)

                # print TABLE
                print('chunk N & mean & std \\\\')
                print('\\toprule')
                for i in range(means.shape[0]):
                    print(f'{i} & {means[i].item():.2f} & {stds[i].item():.2f} \\\\')
                    if (i + 1) < means.shape[0] and (i + 1) % 2 == 0:
                        print('\\midrule')
                print('\\bottomrule')

                # print ACCURACIES
                with torch.no_grad():
                    clean_preds = torch.argmax(attacked_model(cnn_preprocess(x_test.to('cuda:1'))), dim=1)
                    adv_preds = torch.argmax(attacked_model(cnn_preprocess(adv_test.to('cuda:1'))), dim=1)
                    def_preds = torch.argmax(attacked_model(cnn_preprocess(def_test.to('cuda:1'))), dim=1)

                print(f"clean acc: {accuracy(clean_preds.to('cuda:3'), y_test.to('cuda:3'))} - "
                      f"atk acc: {accuracy(adv_preds.to('cuda:3'), y_test.to('cuda:3'))} - "
                      f"def acc {accuracy(def_preds.to('cuda:3'), y_test.to('cuda:3'))}")

                # plot an example
                if metric == 'Linf':
                    test_res = torch.mean(torch.cdist(x_test.to('cuda:2').view(data_n, -1), adv_test.to('cuda:2').view(data_n, -1), p=float('inf')).diag())
                else:
                    test_res = torch.mean(torch.cdist(x_test.to('cuda:2').view(data_n, -1), adv_test.to('cuda:2').view(data_n, -1), p=2).diag())

                plt.imshow(
                    make_grid(
                        torch.cat([x_test[:8].to('cuda:2'), adv_test[:8].to('cuda:2'), def_test[:8].to('cuda:2')], dim=0)
                    ).permute(1, 2, 0).cpu().numpy())
                plt.axis(False)
                plt.suptitle(f"Attacked Model: {model_name} - Attack: {attack_name} with {metric}={bound:.4f}")
                plt.title(f"Metric: {metric} - Bound: {bound:.3f} - Measured: {test_res:.3f}")
                plt.savefig(f"{OUT_FILE}/{model_name}_{attack_name}_{metric}:{bound}.png")


if __name__ == '__main__':

    CKPT_NVAE = '/media/dserez/runs/NVAE/cifar10/best/ours_weighted.pt'
    ADV_BASE_PATH = '/media/dserez/code/adversarial/'
    TORCH_HOME = '/media/dserez/runs/classification/cifar10/'
    OUT_FILE = f'{ADV_BASE_PATH}images/'

    BATCH_SIZE = 100

    # CKPT_NVAE = '/home/dserez/gen_adversarial/loading/ours_weighted.pt'
    # ADV_BASE_PATH = '/home/dserez/gen_adversarial/loading/'
    # TORCH_HOME = '/home/dserez/gen_adversarial/loading/'
    # OUT_FILE = f'{ADV_BASE_PATH}defense_images/'
    #
    # BATCH_SIZE = 200

    main()