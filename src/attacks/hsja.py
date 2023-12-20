# https://github.com/Jianbo-Lab/HSJA/blob/master/hsja.py

import os
import math
import numpy as np
import torch
from kornia.enhance import Normalize
from robustbench import load_cifar10
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def decision_function(model: torch.nn.Module, images: torch.Tensor, params: dict, count_as_query: bool = True):
    """
    :param model: attacked cnn to query
    :param images: images to test
    :param params: clip_min, clip_max, target_label, original_label
    :param count_as_query: if False, do not count "predict" as query.
    :return torch.boolTensor. True = evaluated image[i] as adversarial, False otherwise
    """
    images = torch.clip(images, params['clip_min'], params['clip_max'])
    pred_classes = model.predict(images)
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
    """

    # expand image x1
    b, c, h, w = x2.shape
    x1 = x1.expand(b, c, h, w).view(b, -1)
    x2 = x2.view(b, -1)

    # Compute the distance between two images.
    if constraint == 'l2':
        return torch.mean(torch.cdist(x1, x2, p=2), dim=1).diag()
    elif constraint == 'linf':
        return torch.mean(torch.cdist(x1, x2, float('inf')), dim=1).diag()


def approximate_gradient(model: torch.nn.Module, sample: torch.Tensor, num_evals: int, delta: float, params: dict):
    """
    :param model: attacked CNN for query
    :param sample: src_image
    :param num_evals: number of queries for gradient estimation
    :param delta: param for noise u_b
    :params: clip_max, clip_min, constraint
    """

    clip_max, clip_min = params['clip_max'], params['clip_min']

    # Generate random vectors (u_b for B times).
    _, c, h, w = sample.shape
    B = num_evals

    if params['constraint'] == 'l2':
        rv = torch.normal(mean=torch.zeros((B, c, h, w)), std=torch.ones((B, c, h, w)))
    elif params['constraint'] == 'linf':
        rv = torch.rand(size=(B, c, h, w)) * 2. - 1.  # in range -1, 1
    else:
        raise NotImplementedError(f"constraint: {params['constraint']}")

    # has shape (B, 1, 1, 1)
    normalizer = torch.sqrt(torch.sum(torch.pow(rv, 2), dim=(1, 2, 3), keepdim=True))
    rv = torch.div(rv, normalizer)

    # obtain all perturbed (B, c, h, w)
    perturbed = sample + delta * rv
    perturbed = torch.clip(perturbed, clip_min, clip_max)
    rv = (perturbed - sample) / delta  # done to clip also rv!

    # query the model and obtain signs (B, 1, 1, 1) of values -1, 1.
    decisions = decision_function(model, perturbed, params)  # boolean Tensor of shape B
    final_val = 2 * decisions.view(-1, 1, 1, 1).to(torch.float32) - 1.0

    # Baseline subtraction (when final_val differs / non corner case)
    final_mean = torch.mean(final_val).item()
    if final_mean == 1.0 or final_mean == -1.0:
        # all 1.0 values => label always changes for any noise.
        # all -1.0 values => label never changes for any noise.
        gradf = final_mean * torch.mean(rv, dim=0)
    else:
        final_val -= final_mean  # normalize
        gradf = torch.mean(final_val * rv, dim=0)

    # Get the gradient direction.
    gradf = torch.div(gradf, torch.linalg.norm(gradf)).unsqueeze(0)

    return gradf


def geometric_progression_for_stepsize(x: torch.Tensor, update: torch.Tensor, dist: float,
                                       model: torch.nn.Module, params: dict):
    """
    Geometric progression to search for stepsize.
    Keep decreasing stepsize by half until reaching
    the desired side of the boundary.

    :param x: current sample (1, c, h, w)
    :param update: noise vector found from gradient estimations (moves x in the right direction) (1, c, h, w)
    :param dist: distance from x and the source image
    :param model: attacked cnn for queries
    :param params: cur_iter
    """
    def phi():
        # check if current epsilon moves to a wrong class
        new = x + epsilon * update
        success = decision_function(model, new, params)[0].item()
        return success

    # initialize epsilon
    epsilon = dist / math.sqrt(params['cur_iter'])

    while not phi():
        epsilon /= 2.0

    return epsilon


def select_delta(params, dist_post_update):
    """
    Choose the delta at the scale of distance
    between x and perturbed sample.
    """
    if params['cur_iter'] == 1:
        delta = 0.1 * (params['clip_max'] - params['clip_min'])
    else:
        if params['constraint'] == 'l2':
            delta = np.sqrt(params['d']) * params['theta'] * dist_post_update.item()
        elif params['constraint'] == 'linf':
            delta = params['d'] * params['theta'] * dist_post_update.item()
        else:
            raise NotImplementedError(f"constraint: {params['constraint']}")

    return delta


def binary_search_batch(src_image: torch.Tensor, adv_images: torch.Tensor,
                        model: torch.nn.Module, params: dict):
    """
    Binary search to approach the boundary.
    :param src_image (1, c, h, w)
    :param adv_images (b, c, h, w) (b > 1 only when stepsize_search is grid_search.)
    :param model: attacked model
    :param params: constraint, theta
    :return image on boundary, initial distance from x_src
    """

    def project(alphas):

        alphas = alphas.view(-1, 1, 1, 1)
        if params['constraint'] == 'l2':
            # linear interpolation
            return (1 - alphas) * src_image + alphas * adv_images
        elif params['constraint'] == 'linf':
            # TODO check correctness
            clip_min = src_image - alphas
            clip_max = src_image + alphas
            result = torch.where(adv_images > clip_min, adv_images, clip_min)
            result = torch.where(result < clip_max, result, clip_max)
            return result
        else:
            raise NotImplementedError(f"Unknown Constraint: {params['constraint']}")

    # Compute distance between each x_adv (if more than 1) and x_src.
    initial_dists = compute_distance(src_image, adv_images, params['constraint'])

    # Choose upper thresholds in binary search based on constraint.
    if params['constraint'] == 'linf':
        highs = initial_dists
        thresholds = torch.minimum(initial_dists[0] * params['theta'], params['theta'])
    else:
        highs = torch.ones(adv_images.shape[0])
        thresholds = params['theta']

    lows = torch.zeros(adv_images.shape[0])

    # Start Binary Search Loop.
    while torch.max((highs - lows) / thresholds) > 1:

        # projection to mids.
        mids = (highs + lows) / 2.0
        mid_images = project(alphas=mids)

        # Update highs and lows based on model decisions.
        decisions = decision_function(model, mid_images, params)
        lows = torch.where(torch.logical_not(decisions), mids, lows)
        highs = torch.where(decisions, mids, highs)

    out_images = project(highs)

    # Compute distance of the output image to select the best choice.
    dists = compute_distance(src_image, out_images, params['constraint'])
    idx = torch.argmin(dists)

    dist = initial_dists[idx]
    out_image = out_images[idx].unsqueeze(0)

    return out_image, dist


def initialize(model, _, params):
    """
    :param model: Attacked cnn to query
    :param _: TODO (actually not used, may be removed?)
    :param params: target_image, shape, clip_min, clip_max
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

        # TODO these looks useless, since the step right after initialization is binary search!
        # Binary search to minimize l2 distance to original image.
        # low = 0.0
        # high = 1.0
        # while high - low > 0.001:
        #     mid = (high + low) / 2.0
        #     blended = (1 - mid) * sample + mid * random_noise
        #     success = decision_function(model, blended, params)
        #     if success:
        #         high = mid
        #     else:
        #         low = mid
        #
        # initialization = (1 - high) * sample + high * random_noise
        initialization = random_noise
    else:
        initialization = params['target_image']

    return initialization


def hsja(model: torch.nn.Module, x_src: torch.Tensor, clip_max: int = 1, clip_min: int = 0,
         constraint: str = 'l2', num_iterations: int = 40, gamma: float = 1.0,
         target_label: int = None, target_image: torch.Tensor = None,
         stepsize_search: str = 'geometric_progression', max_num_evals: int = 1e4,
         init_num_evals: int = 100, verbose: bool = True):
    """
    Main algorithm for HopSkipJumpAttack.

    :param model: the object that has predict method.
    :param x_src: src image to attack
    :param clip_max: upper bound of the image.
    :param clip_min: lower bound of the image.
    :param constraint: choose between [l2, linf].
    :param num_iterations: number of iterations.
    :param gamma: used to set binary search threshold theta.
                  The binary search threshold theta is gamma / d^{3/2} for l2 attack
                  and gamma / d^2 for linf attack.
    :param target_label: integer or None for nontargeted attack.
    :param target_image: an array with the same size as sample, or None.
    :param stepsize_search: choose between 'geometric_progression', 'grid_search'.
    :param max_num_evals: maximum number of evaluations for estimating gradient (for each iteration).
                          This is not the total number of model evaluations for the entire algorithm,
                          you need to set a counter of model evaluations by yourself to get that.
                          To increase the total number of model evaluations, set a larger num_iterations.
    :param init_num_evals: initial number of evaluations for estimating gradient.
    :param verbose: print iteration count
    :return perturbed image: torch.Tensor
    """

    # Set parameters
    original_label = model.predict(x_src)[0].item()
    params = {'clip_max': clip_max, 'clip_min': clip_min,
              'shape': x_src.shape,
              'original_label': original_label,
              'target_label': target_label,
              'target_image': target_image,
              'constraint': constraint,
              'num_iterations': num_iterations,
              'gamma': gamma,
              'd': int(np.prod(x_src.shape)),
              'stepsize_search': stepsize_search,
              'max_num_evals': max_num_evals,
              'init_num_evals': init_num_evals,
              'verbose': verbose,
              }

    # Set binary search threshold.
    if params['constraint'] == 'l2':
        params['theta'] = params['gamma'] / (np.sqrt(params['d']) * params['d'])
    else:
        params['theta'] = params['gamma'] / (params['d'] ** 2)

    # Initialize.
    x_adv = initialize(model, x_src, params)

    # Project the initial x_adv to the boundary.
    # dist_pre is distance before binary_search
    # dist_new is distance after binary_search
    x_adv, dist_pre_update = binary_search_batch(x_src, x_adv, model, params)
    dist_new = compute_distance(x_adv, x_src, constraint)

    # start iterations
    for j in range(params['num_iterations']):

        params['cur_iter'] = j + 1

        # Choose delta.
        # delta = select_delta(params, dist_pre_update)
        delta = select_delta(params, dist_new)  # seems to obtain the same result

        # Choose number of evaluations.
        num_evals = int(params['init_num_evals'] * np.sqrt(j + 1))
        num_evals = int(min([num_evals, params['max_num_evals']]))

        # approximate gradient.
        gradf = approximate_gradient(model, x_adv, num_evals, delta, params)

        if params['constraint'] == 'linf':
            update = torch.sign(gradf)
        else:
            update = gradf

        # search step size \epsilon_t.
        if params['stepsize_search'] == 'geometric_progression':

            # find step size.
            epsilon = geometric_progression_for_stepsize(x_adv, update, dist_new, model, params)

            # Update the sample (epsilon ensures it is still adversarial).
            x_adv = torch.clip(x_adv + epsilon * update, clip_min, clip_max)

            # Binary search to return to the boundary.
            x_adv, dist_pre_update = binary_search_batch(x_src, x_adv, model, params)

        elif params['stepsize_search'] == 'grid_search':
            # TODO
            pass
            # # Grid search for stepsize.
            # epsilons = np.logspace(-4, 0, num=20, endpoint=True) * dist
            # epsilons_shape = [20] + len(params['shape']) * [1]
            # perturbeds = perturbed + epsilons.reshape(epsilons_shape) * update
            # perturbeds = clip_image(perturbeds, params['clip_min'], params['clip_max'])
            # idx_perturbed = decision_function(model, perturbeds, params)
            #
            # if np.sum(idx_perturbed) > 0:
            #     # Select the perturbation that yields the minimum distance # after binary search.
            #     perturbed, dist_pre_update = binary_search_batch(sample,
            #                                                       perturbeds[idx_perturbed], model, params)

        # compute new distance.
        dist_new = compute_distance(x_adv, x_src, constraint)
        if verbose:
            print(f'iteration: {j + 1}, {constraint} distance {dist_new.item():.4f}')

        if model.total_queries_count > 7750:
            break

    return x_adv


class AttackedModel(torch.nn.Module):

    def __init__(self, base_model, src_image, constraint='l2'):
        super().__init__()

        self.src_image = src_image
        self.constraint = constraint
        self.base_model = base_model
        self.preprocess = Normalize(mean=torch.tensor([0.507, 0.4865, 0.4409]),
                                    std=torch.tensor([0.2673, 0.2564, 0.2761]))

        # save intermediate distances, adversaries and total query count
        self.save_every = 250
        self.checkpoint = 1
        self.total_queries_count = 0
        self.log = dict()  # key: int = query_count, value: tuple = (Lp distance, x_adv)

    def predict(self, samples: torch.Tensor, count_as_query: bool = True):

        # track progress
        if count_as_query:
            b, _, _, _ = samples.shape
            if b == 1 and self.total_queries_count >= self.checkpoint:
                dist = compute_distance(self.src_image, samples, constraint=self.constraint)
                self.log[self.total_queries_count + 1] = (dist.item(), samples[0].permute(1, 2, 0).cpu().numpy())
                self.checkpoint = self.total_queries_count + self.save_every
            self.total_queries_count += b

        return torch.argmax(self.base_model(self.preprocess(samples)), dim=1)


@torch.no_grad()
def main():

    # Sample to attack
    x_test, y_test = load_cifar10(data_dir='../media/')
    sample = x_test[0].unsqueeze(0)
    target = x_test[250].unsqueeze(0)
    y_target = y_test[250]

    # Attacked CNN
    os.environ["TORCH_HOME"] = '../media/'
    resnet32 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True)
    resnet32 = AttackedModel(resnet32, sample).eval()

    # check that initial label is gt
    y_tested = resnet32.predict(sample, count_as_query=False)
    assert y_test[0] == y_tested[0], "CNN has predicted the wrong class on seen sample!"

    # check that target is predicted correctly
    y_trg_pred = resnet32.predict(target, count_as_query=False)
    assert y_target == y_trg_pred[0], "CNN has predicted the wrong class on seen sample!"

    hsja(resnet32, sample, target_image=target, target_label=y_target)

    # plot results
    log_indices = list(resnet32.log.keys())
    n_logs = len(log_indices) # + 2
    fig, ax = plt.subplots(1 + (n_logs // 8), 8, figsize=(24, 18))

    all_dists = []
    for i, j in enumerate(log_indices):

        dist, x_adv = resnet32.log[j]
        all_dists.append(dist)

        # plot adv
        r = i // 8
        c = i % 8
        ax[r, c].imshow(x_adv)
        ax[r, c].axis(False)
        ax[r, c].set_title(f'queries: {j}')

    ax[-1, -1].imshow(target[0].permute(1, 2, 0).cpu().numpy())
    ax[-1, -1].axis(False)
    ax[-1, -1].set_title(f'x_trg')

    ax[-1, -2].imshow(resnet32.src_image[0].permute(1, 2, 0).cpu().numpy())
    ax[-1, -2].axis(False)
    ax[-1, -2].set_title(f'x_src')

    fig.suptitle('intermediate x_adv and x_src')
    plt.show()
    plt.close(fig)
    plt.plot(log_indices, all_dists)
    plt.xlabel('# queries')
    plt.ylabel('L2 distance')
    plt.show()


if __name__ == '__main__':
    main()
