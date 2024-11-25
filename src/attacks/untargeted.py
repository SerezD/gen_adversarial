import collections
import math

import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import torch as torch
import copy

# TODO CREATE ABSTRACT ATTACK With CALL method.


def l2_norm(x: torch.Tensor, keepdim=False):

    z = (x ** 2).view(-1).sum().sqrt()
    if keepdim:
        z = z.view(1, 1, 1, 1)
    return z


def normalize(x: torch.Tensor):
    """
    :param x: tensor to normalize
    :return: L2 normalized tensor
    """
    return x / l2_norm(x)


class APGDAttack:
    def __init__(self, n_iter: int, rho: float, max_bound: float, ce_loss: bool):
        """
            Implementation of APGD-CE and APGD-DLR attack
            cloned and adapted from https://github.com/fra31/auto-attack/blob/master/autoattack/autopgd_base.py
            uses l2-bound and untargeted setup.

            :param n_iter: number of iterations
            :param rho: parameter for decreasing the step size
            :param max_bound: maximum allowed L2 bound to deem the perturbation successful
            :param ce_loss: whether to use CE (cross entropy) loss or DLR (Difference of Logits Ratio) loss
        """

        self.n_iter = n_iter
        self.rho = rho
        self.max_bound = max_bound

        if ce_loss:
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            self.criterion = self.dlr_loss

        # constant to avoid 0 division error
        self.division_eps = 1e-12

        # parameters to modify step size
        self.initial_step_size_iters = max(int(0.22 * n_iter), 1)
        self.min_step_size_iters = max(int(0.06 * n_iter), 1)
        self.step_size_decr = max(int(0.03 * n_iter), 1)

    def check_loss_oscillation(self, all_losses: torch.Tensor, step: int, lookback: int):
        """
        :param all_losses: all loss values from all iters
        :param step: current step of iterations
        :param lookback: how many values to check in the past from step.
        :return True if loss has not increased at least fixed-threshold times (should change step size)
        """

        # how many loss values to consider?
        prev_losses = all_losses[step - (lookback - 1): step + 1]

        # check number of times loss increased
        shifted_prev_losses = torch.roll(prev_losses, shifts=1, dims=0)
        shifted_prev_losses[0] = prev_losses[0]
        n_loss_incr = torch.gt(prev_losses, shifted_prev_losses).sum().item()

        # if loss is not increasing anymore, then the attack is loosing effectiveness!
        return n_loss_incr < lookback * self.rho

    def dlr_loss(self, logits: torch.Tensor, gt_label: torch.Tensor):
        """
        :param logits: prediction logits, with shape (1, N_Classes)
        :param gt_label: ground truth label of shape (1,)

        :return Loss computed as -(Logit[GT] - Logit[Highest Wrong Prediction]) / (Highest Logit - 3rd Highest Logit)
        """

        _, n = logits.shape

        if n < 4:
            # CHECK https://github.com/fra31/auto-attack/issues/70
            raise AttributeError('APGD_DLR is undefined for problems with less than 4 classes!')

        # sort predictions in ascending order
        logits_sorted, logits_indices_sorted = logits.sort(dim=1)

        # check if attack is successful or not.
        attack_failed = torch.eq(logits_indices_sorted[:, -1], gt_label).item()

        # Compute Loss
        correct_logit = logits[0, gt_label]
        highest_wrong_logit = logits_sorted[:, -2] if attack_failed else logits_sorted[:, -1]
        numerator = -(correct_logit - highest_wrong_logit)  # encourage the highest wrong logit to increase

        highest_logit = logits_sorted[:, -1]

        # normalizer term can lead to vanishing gradient
        # check https://github.com/fra31/auto-attack/issues/108
        if torch.ne(logits_sorted[:, -3], correct_logit):
            normalizer_term = logits_sorted[:, -3]
        else:
            normalizer_term = logits_sorted[:, -4]

        denominator = highest_logit - normalizer_term + self.division_eps

        return numerator / denominator

    def __call__(self, image, gt_label, net):
        """
           :param image: Image of size 1, 3, H, W
           :param gt_label: ground truth label of shape (1,)
           :param net: network (input: images, output: logits -> values of activation **BEFORE** softmax).
        """

        device = image.device

        # get initial adversarial image
        initial_noise = normalize(torch.randn_like(image))

        x_adv = (image + self.max_bound * initial_noise).clamp(0., 1.)

        # save for next round
        x_adv_old = x_adv.clone()

        # get first gradient
        x_adv = x_adv.requires_grad_()
        with torch.enable_grad():
            logits = net(x_adv)
            loss = self.criterion(logits, gt_label)

        grad = torch.autograd.grad(loss, [x_adv])[0].detach()

        # initialize step size parameters
        step_size = 2 * self.max_bound

        # counters
        update_step_size_counter = 0
        step_size_iters = self.initial_step_size_iters

        # update based on previous losses
        loss_steps = torch.zeros([self.n_iter, 1]).to(device)

        # keep track if step size has been decreased previously
        reduced_last_check = True  # start with true ?

        # save best loss and previous best loss to decide if step size should be decreased
        best_loss = loss.item()
        prev_best_loss = loss.item()

        # save best result to restart from there
        x_best = x_adv.clone()
        grad_best = grad.clone()

        # start iterations
        for i in range(self.n_iter):

            # get new adv based on gradient information
            with torch.no_grad():

                # get adv diff
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                # update x_adv
                a = 0.75 if i > 0 else 1.0

                # update using grad
                new_adv = x_adv + step_size * normalize(grad)
                new_adv = normalize(new_adv - image) * torch.min(self.max_bound * torch.ones_like(image).detach(),
                                                                 l2_norm(new_adv - image, keepdim=True))
                new_adv = torch.clamp(image + new_adv, 0., 1.)

                # update using grad 2
                new_adv = x_adv + (new_adv - x_adv) * a + grad2 * (1 - a)
                new_adv = normalize(new_adv - image) * torch.min(self.max_bound * torch.ones_like(image).detach(),
                                                                 l2_norm(new_adv - image, keepdim=True))
                x_adv = torch.clamp(image + new_adv, 0., 1.)

            # get new gradient based on adv loss
            x_adv = x_adv.requires_grad_()
            with torch.enable_grad():
                logits = net(x_adv)
                loss = self.criterion(logits, gt_label)

            grad = torch.autograd.grad(loss, [x_adv])[0].detach()

            # update step size if needed
            with torch.no_grad():

                # append new loss
                loss_steps[i] = loss.item()

                # update params if needed
                if loss.item() > best_loss:
                    best_loss = loss.item()
                    x_best = x_adv.clone()
                    grad_best = grad.clone()

                # check if step size must be updated
                update_step_size_counter += 1
                if update_step_size_counter == step_size_iters:

                    # update step size based on two conditions
                    loss_not_increasing = self.check_loss_oscillation(loss_steps, i, update_step_size_counter)

                    no_improvement = prev_best_loss >= best_loss

                    reduce_step_size = loss_not_increasing or (no_improvement and not reduced_last_check)

                    # update for next round
                    reduced_last_check = reduce_step_size
                    prev_best_loss = best_loss

                    if reduce_step_size:

                        # halve step size and restart from best result.
                        step_size /= 2.0
                        x_adv = x_best.clone()
                        grad = grad_best.clone()

                    # reset counters
                    update_step_size_counter = 0
                    step_size_iters = max(step_size_iters - self.step_size_decr, self.min_step_size_iters)

        # return best guess
        succeed = torch.ne(net(x_adv).argmax(dim=1), gt_label).item()
        bound = torch.linalg.norm((x_adv.detach() - image.detach()).flatten(), ord=2).item()
        return succeed, bound, x_adv.detach()


class AutoAttack:
    def __init__(self):
        """
        Implementation of AutoAttack
        cloned and adapted from https://github.com/fra31/auto-attack/blob/master/autoattack/autoattack.py
        uses l2-bound and untargeted setup.
        """

        # three attacks to test in sequence (Square Attack is not tested)
        # we use different bounds for apgd, since we are interested in minimizing it
        self.apgd_ce1 = APGDAttack(n_iter=64, rho=0.75, max_bound=0.5, ce_loss=True)
        self.apgd_ce2 = APGDAttack(n_iter=64, rho=0.75, max_bound=1.0, ce_loss=True)
        self.apgd_ce3 = APGDAttack(n_iter=64, rho=0.75, max_bound=4.0, ce_loss=True)
        self.apgd_dlr1 = APGDAttack(n_iter=64, rho=0.75, max_bound=0.5, ce_loss=False)
        self.apgd_dlr2 = APGDAttack(n_iter=64, rho=0.75, max_bound=2.0, ce_loss=False)
        self.apgd_dlr3 = APGDAttack(n_iter=64, rho=0.75, max_bound=4.0, ce_loss=False)
        self.fab = FABAttack(n_iter=128, alpha_max=0.1, eta=1.05, beta=0.9)

    def __call__(self, image, gt_label, net):
        """
           :param image: Image of size 1, 3, H, W
           :param gt_label: ground truth label of shape (1,)
           :param net: network (input: images, output: logits -> values of activation **BEFORE** softmax).
        """
        def update_result(s_0, b_0, a_0, s_1, b_1, a_1):

            if s_1 and not s_0:
                return s_1, b_1, a_1
            elif s_1 and s_0:
                if b_1 < b_0:
                    return s_0, b_1, a_1
            return s_0, b_0, a_0

        # apply all attacks, keeping best result

        # APGD-CE (three bounds)
        success, best_bound, best_adv = self.apgd_ce1(image, gt_label, net)

        # test higher bound only if not passed
        if not success:
            s, b, a = self.apgd_ce2(image, gt_label, net)
            success, best_bound, best_adv = update_result(success, best_bound, best_adv, s, b, a)

            if not success:
                s, b, a = self.apgd_ce3(image, gt_label, net)
                success, best_bound, best_adv = update_result(success, best_bound, best_adv, s, b, a)

        # APGD-DLR (three bounds)
        # to apply apgd_dlr check the number of classes
        with torch.no_grad():
            preds = net(image)

        # does not work with less than 3 classes
        if preds.shape[1] > 3:

            # Test three increasing bounds
            s1, b1, a1 = self.apgd_dlr1(image, gt_label, net)

            # test higher bound only if not passed
            if not s1:
                s, b, a = self.apgd_dlr2(image, gt_label, net)
                s1, b1, a1 = update_result(s1, b1, a1, s, b, a)

                if not s1:
                    s, b, a = self.apgd_dlr3(image, gt_label, net)
                    s1, b1, a1 = update_result(s1, b1, a1, s, b, a)

            success, best_bound, best_adv = update_result(success, best_bound, best_adv, s1, b1, a1)

        # FAB attack
        s, b, a = self.fab(image, gt_label, net)
        success, best_bound, best_adv = update_result(success, best_bound, best_adv, s, b, a)

        return success, best_bound, best_adv


class CW:
    def __init__(self, c: float = 1., kappa: float = 0., steps: int = 64, lr: float = 1e-2, n_restarts: int = 1,
                 early_stopping_steps: int = 16):
        """
        cloned and adapted from https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/cw.py

           :param c: parameter for box-constraint
           :param kappa: also written as 'confidence'
           :param steps: number of steps
           :param lr: learning rate of the Adam optimizer.
           :param n_restarts: can restart the iteration from a different random point to avoid falling in local minima
           :param early_stopping_steps: how many loss values in the past to look for deciding early stopping
        """
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.n_restarts = n_restarts

        self.early_stopping_len = early_stopping_steps

    # f-function in the paper
    def f(self, logits, gt_label):

        one_hot = nn.functional.one_hot(gt_label, logits.shape[1])

        real = torch.sum(one_hot * logits, 1)
        other, _ = torch.max((1 - one_hot) * logits - one_hot * 1e4, 1)

        f = torch.max((real - other) + self.kappa, torch.zeros_like(real))
        return f

    def __call__(self, image, label, net):
        """
       :param image: Image of shape (1x3xHxW)
       :param label: ground truth label of shape (1,)
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
        """

        assert len(image.size()) == 4 and image.shape[0] == 1 and len(label.size()) == 1 and label.shape[0] == 1, \
            "wrong input shapes!"

        # original image and label
        image = image.clone().detach()
        label = label.clone().detach()

        absolute_succeed = False
        absolute_best_adv_image = image.clone()
        absolute_best_L2 = 0.

        # initialize loss stuff
        MSELoss = nn.MSELoss(reduction="sum")

        # adaptive c parameter based on restarts
        c = self.c

        # FGSM for initialization, with bound adjusted on image size
        res = np.log2(image.shape[-1])
        initialization_step = FGSM(l2_bound=np.power(2, res-5))

        for _ in range(self.n_restarts):

            # get a random adv image in the direction that increases loss
            best_adv_image = initialization_step(image, label, net)[2]
            noise = torch.randn_like(image)
            noise = noise * np.power(2, res-8) / torch.norm(noise.view(1, -1), dim=1, keepdim=True)
            best_adv_image = torch.clamp(best_adv_image + noise, min=1e-6, max=1 - 1e-6)
            best_L2 = torch.linalg.norm((best_adv_image - image).flatten(), ord=2)

            # initialize w
            w = torch.atanh((best_adv_image * 2.) - 1)
            w.requires_grad = True

            # keep track of losses for early stopping
            rolling_mean_loss = 0.0
            rolling_mean_updates = 0
            prev_succeed = False

            # optimizer
            optimizer = optim.Adam([w], lr=self.lr)

            # start iterations
            for step in range(self.steps):

                # Get adversarial image from w
                current_adv_image = 0.5 * (torch.tanh(w) + 1)

                # Calculate loss
                L2_loss = MSELoss(current_adv_image, image)

                logits = net(current_adv_image)
                f_loss = self.f(logits, label)

                loss = L2_loss + c * f_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([w], max_norm=1.)
                optimizer.step()

                # is attack successful ?
                pred = torch.argmax(logits.detach(), 1)
                succeed = (pred != label).item()

                # succeeding and not converging -> early stopping
                if succeed:

                    # check non convergence
                    loss_item = loss.detach().item()
                    if loss_item > rolling_mean_loss and rolling_mean_updates > self.early_stopping_len:
                        break

                    # update rolling mean
                    lookback = min(rolling_mean_updates, self.early_stopping_len)
                    rolling_mean_loss = (rolling_mean_loss * lookback + loss.item()) / (lookback + 1)
                    rolling_mean_updates += 1

                # update adversarial image if no candidate is found or improved L2
                this_L2 = torch.linalg.norm((current_adv_image.detach() - image).flatten(), ord=2)
                if not prev_succeed or best_L2 > this_L2:
                    best_adv_image = current_adv_image.detach()
                    best_L2 = this_L2
                    prev_succeed = succeed

            # update result if new best is found, depending on current result
            pred = torch.argmax(net(best_adv_image).detach(), 1)
            succeed = (pred != label).item()
            
            # failed at this step -> increase c
            if not succeed:
                c = 1.2 * c
            # found first or better adv at this step -> update and reduce c
            elif (succeed and not absolute_succeed) or (succeed and absolute_succeed and absolute_best_L2 > best_L2):
                c = 0.8 * c
                absolute_best_adv_image = best_adv_image
                absolute_best_L2 = best_L2
                absolute_succeed = True
            # found worse adv at this step -> slightly reduce c
            elif succeed and absolute_succeed and absolute_best_L2 < best_L2:
                c = 0.9 * c

            c = max(min(c, 1000), 0.1)

        return absolute_succeed, absolute_best_L2, absolute_best_adv_image


class DeepFool:

    def __init__(self, num_classes=10, overshoot=0.02, max_iter=50):
        """
        cloned and adapted from https://github.com/LTS4/DeepFool/blob/master/Python/deepfool.py

           :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
           :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
           :param max_iter: maximum number of iterations for deepfool (default = 50)
           :return: success (T/F) - minimal perturbation found - adversarial image
        """

        self.num_classes = num_classes
        self.overshoot = overshoot
        self.max_iter = max_iter

    def zero_gradients(self, x):
        """
        this has been removed from pytorch 1.9.0
        https://discuss.pytorch.org/t/from-torch-autograd-gradcheck-import-zero-gradients/127462/2
        """
        if isinstance(x, torch.Tensor):
            if x.grad is not None:
                x.grad.detach_()
                x.grad.zero_()
        elif isinstance(x, collections.abc.Iterable):
            for elem in x:
                self.zero_gradients(elem)

    def __call__(self, image, gt_label, net):
        """
        cloned and adapted from https://github.com/LTS4/DeepFool/blob/master/Python/deepfool.py

           :param image: Image of size 1xHxWx3
           :param gt_label: truth label of shape (1,)
           :param net: network (input: images, output: values of activation **BEFORE** softmax).
        """

        f_image = net(Variable(image, requires_grad=True)).data.cpu().numpy().flatten()
        I = (np.array(f_image)).flatten().argsort()[::-1]

        I = I[0:self.num_classes]
        label = I[0]

        if gt_label != label:
            # no need to attack a wrong prediction!
            return True, 0.0, image.detach()

        input_shape = tuple(image.shape)
        pert_image = copy.deepcopy(image)
        w = np.zeros(input_shape)
        r_tot = np.zeros(input_shape)

        loop_i = 0

        x = Variable(pert_image, requires_grad=True)
        fs = net(x)
        k_i = label

        while k_i == label and loop_i < self.max_iter:

            pert = np.inf
            fs[0, I[0]].backward(retain_graph=True)
            grad_orig = x.grad.data.cpu().numpy().copy()

            for k in range(1, self.num_classes):
                self.zero_gradients(x)

                fs[0, I[k]].backward(retain_graph=True)
                cur_grad = x.grad.data.cpu().numpy().copy()

                # set new w_k and new f_k
                w_k = cur_grad - grad_orig
                f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

                pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

                # determine which w_k to use
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            r_i = (pert+1e-4) * w / np.linalg.norm(w)
            r_tot = np.float32(r_tot + r_i)

            pert_image = image + (1+self.overshoot)*torch.from_numpy(r_tot).to(image.device)

            x = Variable(pert_image, requires_grad=True)
            fs = net(x)
            k_i = np.argmax(fs.data.cpu().numpy().flatten())

            loop_i += 1

        r_tot = (1+self.overshoot)*r_tot

        # check if succeed or not
        if k_i == gt_label:
            return False, np.inf, image.detach()

        return True, np.linalg.norm(r_tot.flatten(), 2), pert_image.detach()


class FABAttack:
    def __init__(self, n_iter: int, alpha_max: float, eta: float, beta: bool):
        """
            Implementation of FAB attack
            cloned and adapted from https://github.com/fra31/auto-attack/blob/master/autoattack/fab_base.py
            uses l2-bound and untargeted setup.

            :param n_iter: number of iterations
            :param alpha_max: maximum step size
            :param eta: overshoot over the boundary
            :param beta: backward step
        """

        self.n_iter = n_iter
        self.eta = eta
        self.beta = beta
        self.alpha_max = alpha_max

    def get_diff_logits_grads(self, image: torch.Tensor, label: torch.Tensor, net: torch.nn.Module):
        """
        Compute the gradients of the logit difference
        :param image: shape (1, 3, H, W)
        :param label: shape (1, )
        :param net: classifier with logits output
        """
        def zero_gradients(x):
            if isinstance(x, torch.Tensor):
                if x.grad is not None:
                    x.grad.detach_()
                    x.grad.zero_()
            elif isinstance(x, collections.abc.Iterable):
                for elem in x:
                    zero_gradients(elem)

        image = image.clone().requires_grad_()
        with torch.enable_grad():
            y = net(image)

        n_classes = y.shape[1]

        g2 = torch.zeros([n_classes, *image.size()], device=image.device)
        grad_mask = torch.zeros_like(y)
        for i in range(n_classes):
            zero_gradients(image)
            grad_mask[:, i] = 1.0
            y.backward(grad_mask, retain_graph=True)
            grad_mask[:, i] = 0.0
            g2[i] = image.grad.data

        g2 = torch.transpose(g2, 0, 1).detach()
        y2 = y.detach()
        df = y2 - y2[:, label]
        dg = g2 - g2[:, label]
        df[:, label] = 1e10

        return df, dg

    def projection_l2(self, points_to_project, w_hyperplane, b_hyperplane):

        device = points_to_project.device
        t, w, b = points_to_project, w_hyperplane.clone(), b_hyperplane

        c = (w * t).sum(dim=1) - b[:, 0]
        ind2 = 2 * (c >= 0) - 1
        w.mul_(ind2.unsqueeze(1))
        c.mul_(ind2)

        r = torch.max(t / w, (t - 1) / w).clamp(min=-1e12, max=1e12)
        r.masked_fill_(w.abs() < 1e-8, 1e12)
        r[r == -1e12] *= -1
        rs, indr = torch.sort(r, dim=1)
        rs2 = torch.nn.functional.pad(rs[:, 1:], (0, 1))
        rs.masked_fill_(rs == 1e12, 0)
        rs2.masked_fill_(rs2 == 1e12, 0)

        w3s = (w ** 2).gather(1, indr)
        w5 = w3s.sum(dim=1, keepdim=True)
        ws = w5 - torch.cumsum(w3s, dim=1)
        d = -(r * w)
        d.mul_((w.abs() > 1e-8).float())
        s = torch.cat((-w5 * rs[:, 0:1], torch.cumsum((-rs2 + rs) * ws, dim=1) - w5 * rs[:, 0:1]), 1)

        c4 = s[:, 0] + c < 0
        c3 = (d * w).sum(dim=1) + c > 0
        c2 = ~(c4 | c3)

        lb = torch.zeros(c2.sum(), device=device)
        ub = torch.full_like(lb, w.shape[1] - 1)
        nitermax = math.ceil(math.log2(w.shape[1]))

        s_, c_ = s[c2], c[c2]
        for counter in range(nitermax):
            counter4 = torch.floor((lb + ub) / 2)
            counter2 = counter4.long().unsqueeze(1)
            c3 = s_.gather(1, counter2).squeeze(1) + c_ > 0
            lb = torch.where(c3, counter4, lb)
            ub = torch.where(c3, ub, counter4)

        lb = lb.long()

        if c4.any():
            alpha = c[c4] / w5[c4].squeeze(-1)
            d[c4] = -alpha.unsqueeze(-1) * w[c4]

        if c2.any():
            alpha = (s[c2, lb] + c[c2]) / ws[c2, lb] + rs[c2, lb]
            alpha[ws[c2, lb] == 0] = 0
            c5 = (alpha.unsqueeze(-1) > r[c2]).float()
            d[c2] = d[c2] * c5 - alpha.unsqueeze(-1) * w[c2] * (1 - c5)

        return d * (w.abs() > 1e-8).float()

    def __call__(self, image, gt_label, net):
        """
           :param image: Image of size 1, 3, H, W
           :param gt_label: ground truth label of shape (1,)
           :param net: network (input: images, output: logits -> values of activation **BEFORE** softmax).
        """

        device = image.device
        image = image.detach().clone()

        # check pred
        with torch.no_grad():
            pred = torch.argmax(net(image))
        if pred != gt_label:
            return True, 0.0, image.detach()

        # initialization
        x_adv = image.clone()
        bound = 1e10
        succeed = False

        x_orig = image.clone()
        x_i = image.clone()  # first iteration is x_orig
        x_orig_flat = image.clone().view(1, -1)

        for _ in range(self.n_iter):

            with torch.no_grad():

                # get class s with the decision hyperplane closest to current adv
                # (Equation 7 of the paper)
                df, dg = self.get_diff_logits_grads(x_i, gt_label, net)
                dist = df.abs() / (1e-12 + (dg ** 2).reshape(1, df.shape[1], -1).sum(dim=-1).sqrt())
                closest_class_index = dist.min(dim=1).indices

                # projections
                dg2 = dg[:, closest_class_index]
                b = - df[:, closest_class_index] + (dg2 * x_i).view(1, -1).sum(dim=-1)
                w = dg2.view([1, -1])

                # these are both projections
                d3 = self.projection_l2(
                    torch.cat((x_i.view(1, -1), x_orig_flat), 0),
                    torch.cat((w, w), 0),
                    torch.cat((b, b), 0))

                # these are di, d_orig
                d1 = torch.reshape(d3[:1], x_i.shape)
                d2 = torch.reshape(d3[-1:], x_i.shape)

                # get alpha (equation 9)
                a0 = (d3 ** 2).sum(dim=1, keepdim=True).sqrt().view(-1, 1, 1, 1)
                a0 = torch.max(a0, 1e-8 * torch.ones_like(a0))
                a1 = a0[:1]
                a2 = a0[-1:]

                alpha = torch.max(a1 / (a1 + a2), torch.zeros_like(a1))
                alpha = torch.min(alpha, self.alpha_max * torch.ones_like(a1))

                # update x_i
                x_i = ((x_i + self.eta * d1) * (1 - alpha) + (x_orig + d2 * self.eta) * alpha).clamp(0.0, 1.0)

                # update adv, min_bound if x_i is adv
                succeed_i = torch.ne(net(x_i).argmax(dim=1), gt_label).item()
                if succeed_i:
                    succeed = True
                    t = ((x_i - x_orig) ** 2).view(1, -1).sum(dim=-1).sqrt().item()
                    if t < bound:
                        x_adv = x_i.clone()
                        bound = t
                    x_i = (1 - self.beta) * x_orig + self.beta * x_i

        # return best guess
        return succeed, bound, x_adv.detach()


class FGSM:
    def __init__(self, l2_bound: float):
        """
        cloned and adapted from https://github.com/1Konny/FGSM/blob/master/adversary.py
        """
        self.l2_bound = l2_bound

    def __call__(self, image, label, net):
        """
        Returns:
            succeed (T/F) - minimal perturbation found - adversarial image
        """
        x_adv = Variable(image.data, requires_grad=True)
        h_adv = net(x_adv)

        # check if pred is correct.
        if torch.argmax(h_adv, dim=-1) != label:
            return True, 0.0, image

        cost = -torch.nn.functional.cross_entropy(h_adv, label)

        net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()

        # Normalize the gradient to unit norm
        perturbation = x_adv.grad
        perturbation_norm = torch.norm(perturbation.view(perturbation.size(0), -1), p=2, dim=1, keepdim=True)
        perturbation = perturbation / perturbation_norm

        # project to L2 norm ball
        x_adv = x_adv - perturbation * self.l2_bound
        x_adv = torch.clamp(x_adv, 0., 1.)

        # check prediction
        h_adv = net(x_adv)

        return torch.argmax(h_adv, dim=-1) != label, self.l2_bound, x_adv.detach()
