"""
take different autoattack attacks.
encode adversaries with nvae and check distance from origin w.r.t clean
"""
import math

import torch
import os
import time

from einops import pack
from kornia.enhance import Normalize
from matplotlib import pyplot as plt
from torchmetrics import Accuracy
from tqdm import tqdm

from autoattack.autopgd_base import APGDAttack
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

    cnn_preprocess = Normalize(mean=torch.tensor([0.507, 0.4865, 0.4409], device='cuda:1'),
                               std=torch.tensor([0.2673, 0.2564, 0.2761], device='cuda:1'))
    predict_pass = lambda x: resnet32(cnn_preprocess(x))
    accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1).to('cuda:0')

    metric, bound = 'L2', 0.5
    attack = APGDAttack(predict_pass, norm=metric, eps=bound)

    # 2. TEST CHUNK SHIFT after attack

    # measured_chunks_diff_cln_adv = torch.empty((0, sum(nvae.groups_per_scale)), device='cuda:1')
    #
    # for b in tqdm(range(0, data_n, BATCH_SIZE)):
    #
    #     x = x_test[b:b + BATCH_SIZE]
    #     y = y_test[b: b + BATCH_SIZE]
    #
    #     # compute x_adv
    #     x_adv = attack.perturb(x.to('cuda:1'), y.to('cuda:1'))
    #
    #     # encode and get chunks
    #     chunks_cln, cln_mu_x, cln_mu_z, cln_sampled_mu_x, cln_sampled_mu_z = nvae.encode(x.to('cuda:0'))
    #     chunks_adv, adv_mu_x, adv_mu_z, adv_sampled_mu_x, adv_sampled_mu_z = nvae.encode(x_adv.to('cuda:0'))
    #
    #     # mu x + mu z
    #     # chunks_cln = [mux + muz for (mux, muz) in zip(cln_mu_x, cln_mu_z)]
    #     # chunks_adv = [mux + muz for (mux, muz) in zip(adv_mu_x, adv_mu_z)]
    #
    #     # l2 distance CLN ADV
    #     chunks_l2 = torch.empty((BATCH_SIZE, 0), device='cuda:1')
    #     for (c, a) in zip(chunks_cln, chunks_adv):
    #         chunks_l2, _ = pack([
    #             chunks_l2,
    #             (c.to('cuda:1').view(BATCH_SIZE, -1) - a.to('cuda:1').view(BATCH_SIZE, -1)).pow(2).sum(dim=1).sqrt().unsqueeze(1)
    #         ] , 'b *')
    #
    #     measured_chunks_diff_cln_adv, _ = pack([measured_chunks_diff_cln_adv, chunks_l2], '* n')
    #
    # l2_means = torch.mean(measured_chunks_diff_cln_adv, dim=0)
    # l2_stds = torch.std(measured_chunks_diff_cln_adv, dim=0)
    #
    # # print TABLE
    # print('chunk N & mean & std \\\\')
    # print('\\toprule')
    # for i in range(l2_means.shape[0]):
    #     print(f'{i} & {l2_means[i].item():.2f} & {l2_stds[i].item():.2f} \\\\')
    #     if (i + 1) < l2_means.shape[0] and (i + 1) % 2 == 0:
    #         print('\\midrule')
    # print('\\bottomrule')

    # 3. CLEAN by sampling.
    n = 8
    x_adv_test = torch.empty((0, 3, 32, 32), device='cuda:1')
    x_def_test = torch.empty((0, n, 3, 32, 32), device='cuda:1')
    x_cln_test = torch.empty((0, n, 3, 32, 32), device='cuda:1')

    for b in tqdm(range(0, data_n, BATCH_SIZE)):

        x = x_test[b:b + BATCH_SIZE]
        y = y_test[b: b + BATCH_SIZE]

        # compute x_adv
        x_adv = attack.perturb(x.to('cuda:1'), y.to('cuda:1'))
        x_adv_test, _ = pack([x_adv_test, x_adv], '* c h w')

        x_def_tmp = torch.empty((BATCH_SIZE, 0, 3, 32, 32), device='cuda:1')
        x_cln_tmp = torch.empty((BATCH_SIZE, 0, 3, 32, 32), device='cuda:1')
        for i in range(n):

            # encode adv and get chunks
            chunks_adv, _, _, _, _ = nvae.encode(x_adv.to('cuda:0'))
            x_def = nvae.decoder_output(nvae.sample_from_chunk0(chunks_adv[:2], t=1.)).mean()
            x_def_tmp, _ = pack([x_def_tmp, x_def.to('cuda:1').unsqueeze(1)], 'b * c h w')

            # encode adv and get chunks
            chunks_cln, _, _, _, _ = nvae.encode(x.to('cuda:0'))
            x_cln = nvae.decoder_output(nvae.sample_from_chunk0(chunks_cln[:2], t=1.)).mean()
            x_cln_tmp, _ = pack([x_cln_tmp, x_cln.to('cuda:1').unsqueeze(1)], 'b * c h w')

        x_def_test, _ = pack([x_def_test, x_def_tmp], '* n c h w')
        x_cln_test, _ = pack([x_cln_test, x_cln_tmp], '* n c h w')

    clean_preds = torch.argmax(resnet32(cnn_preprocess(x_test.to('cuda:1'))), dim=1)
    adv_preds = torch.argmax(resnet32(cnn_preprocess(x_adv_test.to('cuda:1'))), dim=1)

    def_preds = torch.empty((data_n, n), device='cuda:1')
    clnd_preds = torch.empty((data_n, n), device='cuda:1')
    for i in range(n):
        def_preds[:, i] = torch.argmax(resnet32(cnn_preprocess(x_def_test[:, i])), dim=1)
        clnd_preds[:, i] = torch.argmax(resnet32(cnn_preprocess(x_cln_test[:, i])), dim=1)

    def_preds = torch.mode(def_preds, dim=1).values
    clnd_preds = torch.mode(clnd_preds, dim=1).values

    print(f"clean acc: {accuracy(clean_preds.to('cuda:0'), y_test.to('cuda:0')):.3f} - "
          f"clean def acc: {accuracy(clnd_preds.to('cuda:0'), y_test.to('cuda:0')):.3f} - "
          f"atk acc: {accuracy(adv_preds.to('cuda:0'), y_test.to('cuda:0')):.3f} - "
          f"def acc: {accuracy(def_preds.to('cuda:0'), y_test.to('cuda:0')):.3f} ")

    tensors = [x_test[:8].to('cuda:1'), x_cln_test[:8, 0].to('cuda:1'),
               x_adv_test[:8].to('cuda:1'), x_def_test[:8, 0].to('cuda:1')]

    plt.imshow(make_grid(torch.cat(tensors, dim=0)).permute(1, 2, 0).cpu().numpy())
    plt.axis(False)
    plt.show()


if __name__ == '__main__':

    CKPT_NVAE = '/media/dserez/runs/NVAE/cifar10/best/3scales_1group_NF.pt'
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