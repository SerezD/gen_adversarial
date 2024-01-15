import os
import math
import numpy as np
import torch
from einops import pack
from kornia.enhance import Normalize
from torch.utils.data import DataLoader

from data.datasets import CoupledDataset

from matplotlib import pyplot as plt

from src.NVAE.original.model import AutoEncoder
from src.NVAE.original.utils import get_arch_cells


# hsja but:
# binary search interpolation in the latent space. (low level)
# gradient estimation in the latent space (u_b) are latent vectors. (high level)


class NVAEModel(torch.nn.Module):

    def __init__(self, base_model: torch.nn.Module, device: str = 'cuda:0'):

        super().__init__()

        self.device = device
        self.base_model = base_model.to(self.device).eval()

    def encode(self, images: torch.Tensor):
        """
        :return list of chunks
        """
        return self.base_model.encode_deterministic(images.to(self.device))

    def decode(self, chunks: list):
        """
        reconstructed images (b, c, h, w) from chunks
        """
        logits = self.base_model.decode(chunks)
        return self.base_model.decoder_output(logits).mean()


class AttackedModel(torch.nn.Module):

    def __init__(self, base_model: torch.nn.Module, src_image: torch.Tensor, constraint: str = 'l2',
                 max_queries: int = 10000, device: str = 'cuda:0'):

        super().__init__()

        self.device = device
        self.constraint = constraint
        self.max_queries = max_queries

        self.base_model = base_model.to(device)
        self.src_image = src_image.to(device)

        self.preprocess = Normalize(mean=torch.tensor([0.507, 0.4865, 0.4409]),
                                    std=torch.tensor([0.2673, 0.2564, 0.2761]))

        # save query_count / lp distance
        self.log = torch.empty((0, 2), device=device)

    def predict(self, samples: torch.Tensor, count_as_query: bool = True):

        samples = samples.to(self.device)

        # track progress
        if count_as_query:

            b, _, _, _ = samples.shape
            dist = compute_distance(self.src_image, samples, constraint=self.constraint)

            if self.log.shape[0] == 0:  # is log empty ?
                queries = torch.arange(0, b, device=self.device)
            else:
                actual_count = int(self.log[-1, 0]) + 1
                queries = torch.arange(actual_count, actual_count + b, 1, device=self.device)

            self.log, _ = pack([self.log, torch.stack([queries, dist], dim=1)], '* n')

        return torch.argmax(self.base_model(self.preprocess(samples)), dim=1)


def decision_function(model: AttackedModel, images: torch.Tensor, params: dict, count_as_query: bool = True):
    """
    :param model: attacked cnn to query
    :param images: images to test (b, c, h, w)
    :param params: clip_min: int, clip_max: int, target_label: int, original_label: int
    :param count_as_query: if False, do not count "predict" as query.
    :return torch.boolTensor of shape b. True = evaluated image[i] as adversarial, False otherwise
    """

    images = torch.clip(images, params['clip_min'], params['clip_max'])

    pred_classes = model.predict(images, count_as_query=count_as_query)

    if params['target_label'] is None:
        return torch.ne(pred_classes, params['original_label'])
    else:
        return torch.eq(pred_classes, params['target_label'])


def compute_distance(x1: torch.Tensor, x2: torch.Tensor, constraint='l2'):
    """
    Compute Lp distance where p is 2, inf.
    :param x1: (1 c h w)
    :param x2: (b c h w)
    :param constraint: l2 or linf
    :return lp distance: torch.Tensor of shape (b)
    """

    # expand image x1
    b, c, h, w = x2.shape
    x1 = x1.expand(b, c, h, w).view(b, -1)
    x2 = x2.view(b, -1)

    # Compute the distance between images.
    return torch.cdist(x1, x2, p=2 if constraint == 'l2' else float('inf')).diag()


def approximate_gradient(model: AttackedModel, nvae, sample: torch.Tensor, num_evals: int, delta: float, params: dict):
    """
    :param model: attacked CNN for query
    :param sample: src_image (1, c, h, w)
    :param num_evals: max number of queries for gradient estimation
    :param delta: param for noise u_b
    :param params: clip_max: float, clip_min: float, constraint: str 'l2' or 'linf'
    :return: estimated gradient direction (eq. 9 of the paper)
    """

    # gradient is approximated with the formula:
    #   1/B \sum_b=1^B sign(predict(x + \delta * u_b)) * u_b

    clip_max, clip_min = params['clip_max'], params['clip_min']
    _, C, H, W = sample.shape
    device = sample.device

    # 1. Generate random vectors (u_b for num_evals times, where num_evals is B in the formula).
    #    use gaussian noise for l2 and uniform noise for linf
    if params['constraint'] == 'l2':
        chunks = nvae.encode(sample)
        u_noise = []
        for ck in chunks:
            _, d = ck.shape
            u_noise.append(
                torch.normal(mean=torch.zeros((num_evals, d)), std=torch.ones((num_evals, d))).to(device)
            )
    elif params['constraint'] == 'linf':
        u_noise = torch.rand(size=(num_evals, C, H, W), device=device) * 2. - 1.  # in range -1, 1
    else:
        raise NotImplementedError(f"constraint: {params['constraint']}")

    # Each noise vector is also normalized
    for i, u in enumerate(u_noise.copy()):
        denominator = torch.sqrt(torch.sum(torch.pow(u, 2), dim=1, keepdim=True))
        u_noise[i] = torch.div(u, denominator)

    # 2. obtain all perturbed samples to test
    sample_chunks = nvae.encode(sample)
    perturbed_chunks = []
    for i, u in enumerate(u_noise):
        perturbed_chunks.append(sample_chunks[i] + 0.01 * u)

    perturbed_samples = torch.clip(nvae.decode(perturbed_chunks.copy()), clip_min, clip_max)

    # clip also each u_b
    u_noise = (perturbed_samples - sample) # / delta

    # 3. query the model and obtain sign(predict(x + \delta * u_b)) = -1 or 1.
    decisions = decision_function(model, perturbed_samples, params).view(-1, 1, 1, 1).to(torch.float32)
    signs = 2 * decisions - 1.0

    # 4. multiply each u_b by its sign and take mean
    # In practice, u_b is also scaled (normalized) by a factor depending on how many correct decisions appear
    signs_mean = torch.mean(signs).item()

    if signs_mean == 1.0 or signs_mean == -1.0:
        # corner cases (don't need to normalize)
        # all 1.0 values => label always changes for any noise.
        # all -1.0 values => label never changes for any noise.
        gradf = signs_mean * torch.mean(u_noise, dim=0).unsqueeze(0)
    else:
        signs -= signs_mean  # normalize
        gradf = torch.mean(signs * u_noise, dim=0).unsqueeze(0)

    # 5. Get the final gradient direction.
    gradf = torch.div(gradf, torch.linalg.norm(gradf))

    return gradf if params['constraint'] == 'l2' else torch.sign(gradf)


def geometric_progression_for_stepsize(x: torch.Tensor, update: torch.Tensor, dist: float,
                                       model: AttackedModel, params: dict):
    """
    search stepsize epsilon such that predict(x + epsilon * update) = y_trg.
    Keeps decreasing epsilon by half until reaching the desired side of the boundary.

    :param x: current sample (1, c, h, w)
    :param update: noise vector found from gradient estimation (moves x in the right direction) (1, c, h, w)
    :param dist: distance from x and the source image
    :param model: attacked cnn for obtaining predictions
    :param params: needs cur_iter: int
    """
    def phi():
        # check if current epsilon moves to a wrong class
        candidate_adv = torch.clip(x + epsilon * update, params['clip_min'], params['clip_max'])
        return decision_function(model, candidate_adv, params, count_as_query=True)[0].item()

    # initialize epsilon
    epsilon = dist / math.sqrt(params['cur_iter'])

    while not phi():
        epsilon /= 2.0

    return epsilon


def select_delta(params: dict, dist: float):
    """
    Choose the delta at the scale of distance between x and perturbed sample.
    """
    if params['cur_iter'] == 1:
        delta = 0.1 * (params['clip_max'] - params['clip_min'])
    else:
        if params['constraint'] == 'l2':
            delta = np.sqrt(params['d']) * params['theta'] * dist
        elif params['constraint'] == 'linf':
            delta = params['d'] * params['theta'] * dist
        else:
            raise NotImplementedError(f"constraint: {params['constraint']}")

    return delta


def latent_binary_search(src_image: torch.Tensor, adv_image: torch.Tensor,
                         src_chunks: list, adv_chunks: list, model: AttackedModel, nvae: NVAEModel,
                         params: dict):
    """
    Binary search to approach the boundary.
    :param src_image (1, c, h, w)
    :param adv_image (1, c, h, w)
    :param model: attacked model
    :param nvae: NVAEModel
    :param params: constraint, theta
    :return new adv_image on  (1, c, h, w), distance (adv_image, x_src): float
    """

    def project(alpha):

        if params['constraint'] == 'l2':
            # linear interpolation on chunks

            # tmp = [(1 - alpha) * s + alpha * a for (s, a) in zip(src_chunks, adv_chunks)]
            # tmp[2] = adv_chunks[2]
            # return tmp

            # return [(1 - alpha) * s + alpha * a for (s, a) in zip(src_chunks, adv_chunks)]
            return (1 - alpha) * src_image + alpha * adv_image

        elif params['constraint'] == 'linf':
            # TODO what happens with linf ?
            clip_min = src_image - alpha
            clip_max = src_image + alpha
            result = torch.where(adv_image > clip_min, adv_image, clip_min)
            result = torch.where(result < clip_max, result, clip_max)
            return result
        else:
            raise NotImplementedError(f"Unknown Constraint: {params['constraint']}")

    # Choose upper threshold in binary search based on constraint.
    if params['constraint'] == 'linf':
        # TODO
        # Compute distance between x_adv and x_src.
        initial_dist = compute_distance(src_image, adv_image, params['constraint'])[0].item()

        high = initial_dist
        threshold = torch.minimum(initial_dist * params['theta'], params['theta'])
    else:
        high = 1.
        threshold = params['theta']  # between 0 and 1

    low = 0

    # Start Binary Search Loop.
    out_image = None
    while (high - low) / threshold > 1:

        # projection to mid.
        mid = (high + low) / 2.0

        mid_image = project(alpha=mid)
        # mid_image = nvae.decode(project(alpha=mid))

        # Update high and low based on model decision
        is_adversarial = decision_function(model, mid_image, params, count_as_query=True)[0].item()  # true or false
        if is_adversarial:
            high = mid  # next interpolation more like src, less like adv
            out_image = mid_image
        else:
            low = mid  # next interpolation more like adv, less like src

    # final projection
    if out_image is None:
        # test
        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for alpha in alphas:
            sample = [(1 - alpha) * s + alpha * a for (s, a) in zip(src_chunks, adv_chunks)]
            sample = nvae.decode(sample)
            # sample = (1 - alpha) * src_image + alpha * adv_image
            print(f"alpha: {alpha} - is_adv: {decision_function(model, sample, params, count_as_query=False)[0].item()}")
        print("#" * 20)
        raise RuntimeError("adv_image is not adversarial!")

    # Compute distance of the output image
    dist = compute_distance(src_image, out_image, params['constraint'])[0].item()

    # is_old_adv = decision_function(model, adv_image, params, count_as_query=False)[0].item()
    # is_new_adv = decision_function(model, out_image, params, count_as_query=False)[0].item()
    # from torchvision.utils import make_grid
    # plt.imshow(make_grid(torch.cat([src_image, adv_image, out_image], dim=0), nrow=3).permute(1, 2, 0).cpu().numpy())
    # plt.axis(False)
    # plt.title(f"x_src.  old_adv.   new_adv. \n DIST: {dist:.4f} IS_OLD_ADV: {is_old_adv}. IS_NEW_ADV: {is_new_adv}")
    # plt.show()

    return out_image, dist


def initialize(model: AttackedModel, params: dict):
    """
    :param model: Attacked cnn to query
    :param params: target_image: torch.Tensor (1, c, h, w), shape: tuple (b, c, h, w), clip_min: float, clip_max: float
    :return initial adversarial sample (target image if specified, else a random noise vector)
    """

    num_evals = 0

    if params['target_image'] is None:

        # Find a misclassified random noise.
        while True:

            # sample noisy image from uniform distribution [0, 1).
            random_noise = torch.rand(size=params['shape'])

            # normalize in range clip_min, clip_max
            ones = torch.ones_like(random_noise)
            random_noise = torch.div(random_noise - ones * params['clip_min'],
                                     ones * params['clip_max'] - ones * params['clip_min'])

            # not counted as query
            success = decision_function(model, random_noise, params, count_as_query=False)[0].item()
            num_evals += 1
            if success:
                break
            assert num_evals < 1e4, "Initialization failed! Use a misclassified image as `target_image`"

        initialization = random_noise
    else:
        initialization = params['target_image']

    return initialization


def latent_hsja(model: AttackedModel, nvae: NVAEModel, x_src: torch.Tensor, original_label: int, clip_max: float = 1.,
                clip_min: float = 0., constraint: str = 'l2', num_iterations: int = 32, gamma: float = 1.0,
                target_label: int = None, target_image: torch.Tensor = None, max_num_evals: int = 1e4,
                init_num_evals: int = 100, verbose: bool = True):
    """
    HopSkipJumpAttack with Latent NVAE model.

    :param model: the object that has predict method.
    :param nvae: the object gives you chunks (encode) and reconstructions (decode).
    :param x_src: src image to attack
    :param original_label: int
    :param clip_max: upper bound of the image.
    :param clip_min: lower bound of the image.
    :param constraint: choose between [l2, linf].
    :param num_iterations: number of iterations (project on boundary, find gradient, move).
    :param gamma: used to set binary search threshold theta.
                  The binary search threshold theta is gamma / d^{3/2} for l2 attack
                  and gamma / d^2 for linf attack.
    :param target_label: integer or None for nontargeted attack.
    :param target_image: an array with the same size as sample, or None.
    :param max_num_evals: maximum number of evaluations when estimating gradient (in each iteration).
    :param init_num_evals: initial number of evaluations when estimating gradient (at very first iteration).
    :param verbose: print iteration count
    :return perturbed image: torch.Tensor (1, c, h, w)
    """

    # Set parameters
    params = {'clip_max': clip_max, 'clip_min': clip_min,
              'shape': x_src.shape,
              'original_label': original_label,
              'target_label': target_label,
              'target_image': target_image,
              'constraint': constraint,
              'd': int(np.prod(x_src.shape)),
              'max_num_evals': max_num_evals,
              'init_num_evals': init_num_evals,
              'verbose': verbose,
              }

    # Set binary search threshold.
    if params['constraint'] == 'l2':
        params['theta'] = gamma / (np.sqrt(params['d']) * params['d'])
    else:
        params['theta'] = gamma / (params['d'] ** 2)

    # Initialize.
    x_adv = initialize(model, params)

    # get chunk only once (ensure determinism!)
    src_chunks = nvae.encode(x_src)
    adv_chunks = nvae.encode(x_adv)

    # start iterations
    for j in range(num_iterations):

        params['cur_iter'] = j + 1

        # Project x_adv to the boundary.
        # dist is distance(x_src, x_adv)
        x_adv, dist = latent_binary_search(x_src, x_adv, src_chunks, adv_chunks, model, nvae, params)

        # Choose delta.
        delta = select_delta(params, dist)

        # Choose number of evaluations for gradient estimation.
        num_evals = int(params['init_num_evals'] * np.sqrt(j + 1))
        num_evals = int(min([num_evals, params['max_num_evals']]))

        # find noise image in correct direction (1, c, h, w)
        update = approximate_gradient(model, nvae, x_adv, num_evals, delta, params)

        # search step size = \epsilon_t.
        epsilon = geometric_progression_for_stepsize(x_adv, update, dist, model, params)

        # Update the sample (epsilon ensures it is still adversarial).
        x_adv = torch.clip(x_adv + epsilon * update, clip_min, clip_max)

        adv_chunks = nvae.encode(x_adv)

        # ensure x_adv is still adversarial when autoencode
        if not decision_function(model, nvae.decode(nvae.encode(x_adv)), params, count_as_query=False)[0].item():
            # stop here
            break

        if model.log.shape[0] >= model.max_queries:
            break

        if verbose:
            print(f'iteration: {j + 1}, {constraint} distance {dist:.4f}')

    return x_adv


@torch.no_grad()
def main(data_dir: str, model_dir: str, nvae_checkpoint: str):

    # data
    dataloader = DataLoader(CoupledDataset(folder=data_dir, image_size=32), batch_size=1, shuffle=False)

    # Attacked CNN
    os.environ["TORCH_HOME"] = model_dir
    base_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True)

    # NVAE
    # load nvae pretrained cifar10
    checkpoint = torch.load(nvae_checkpoint, map_location='cpu')

    # get and update args
    args = checkpoint['args']
    args.num_mixture_dec = 10

    # init model and load
    arch_instance = get_arch_cells(args.arch_instance)
    nvae = AutoEncoder(args, None, arch_instance)
    nvae.load_state_dict(checkpoint['state_dict'], strict=False)
    nvae = NVAEModel(nvae, device='cuda:0')

    # params
    constraint = 'l2'
    num_steps = 100
    step = 0

    # final_log (save first 10000 queries for each tested image)
    n_queries = 2500
    final_log = torch.empty((0, n_queries, 2), device="cuda:0")

    for (src_x, src_y, trg_x, trg_y) in dataloader:

        print(f"{step}/{num_steps}")

        resnet32 = AttackedModel(base_model.eval(), src_x, constraint=constraint, max_queries=n_queries).eval()

        # check that src, target are predicted correctly
        pred_src = resnet32.predict(src_x, count_as_query=False)
        pred_trg = resnet32.predict(trg_x, count_as_query=False)

        # pred_src = resnet32.predict(nvae.decode(nvae.encode(src_x)), count_as_query=False)
        # pred_trg = resnet32.predict(nvae.decode(nvae.encode(trg_x)), count_as_query=False)

        if pred_src[0] != src_y[0] or pred_trg[0] != trg_y[0]:
            print("wrong label predicted! skipping sample!")
            continue

        latent_hsja(resnet32, nvae, src_x.to("cuda:0"), original_label=src_y.item(), constraint=constraint,
                    target_image=trg_x.to("cuda:0"), target_label=trg_y.item(), verbose=False)

        if resnet32.log.shape[0] < n_queries:
            print("could not be attacked!")
            continue

        final_log, _ = pack([final_log, resnet32.log[:n_queries].unsqueeze(0)], '* n d')

        step += 1
        if step == num_steps:
            break

    x_axis = np.arange(n_queries)
    final_log_mean = torch.mean(final_log[:, :, 1], dim=0).cpu().numpy()
    final_log_max = torch.max(final_log[:, :, 1], dim=0).values.cpu().numpy()
    final_log_min = torch.min(final_log[:, :, 1], dim=0).values.cpu().numpy()

    # Plotting
    plt.plot(x_axis, final_log_mean, color='red')
    plt.fill_between(x_axis, final_log_min, final_log_max, color='pink', alpha=0.3)
    plt.xlabel("# queries")
    plt.ylabel("distance")
    plt.ylim(0., 5.)
    plt.title(f"Targeted HSJA. Constraint = {constraint} ({num_steps} images of Cifar10, ResNet)")
    plt.savefig(f"./figures/lhsja-ge_targeted_distance={constraint}_steps={num_steps}.png")
    plt.close()


if __name__ == '__main__':
    main(data_dir="/media/dserez/datasets/cifar10/validation/", model_dir="/media/dserez/runs/adversarial/CNNs/",
         nvae_checkpoint="/media/dserez/runs/NVAE/cifar10/best/3scales_1group.pt")
