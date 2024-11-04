import collections

import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import torch as torch
import copy


class CW:
    def __init__(self, c: float = 1., kappa: float = 0., steps: int = 64, lr: float = 1e-2):
        """
        cloned and adapted from https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/cw.py

           :param c: parameter for box-constraint
           :param kappa: also written as 'confidence'
           :param steps: number of steps
           :param lr: learning rate of the Adam optimizer.
           :param amp: if true, use Automatic Mixed Precision
        """
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr

    # f-function in the paper
    def f(self, preds, labels):

        one_hot_labels = torch.eye(preds.shape[1], device=preds.device)[labels]

        # find the max logit other than the target class
        other = torch.max((1 - one_hot_labels) * preds, dim=1)[0]

        # get the target class's logit
        real = torch.max(one_hot_labels * preds, dim=1)[0]

        return torch.clamp((real - other), min=-self.kappa)

    def __call__(self, image, label, net):
        """
       :param image: Image of size HxWx3
       :param ground truth label
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
        """

        images = image.clone().detach()
        labels = label.clone().detach()

        w = torch.atanh(torch.clamp(images * 2 - 1, min=-1, max=1)).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).to(images.device)
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction="none")
        Flatten = nn.Flatten()

        optimizer = optim.Adam([w], lr=self.lr)

        for step in range(self.steps):

            # Get adversarial images
            adv_images = 1 / 2 * (torch.tanh(w) + 1)

            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_images), Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = net(adv_images)
            f_loss = self.f(outputs, labels).sum()

            cost = L2_loss + self.c * f_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update adversarial images
            pre = torch.argmax(outputs.detach(), 1)

            # If the attack is not targeted we simply make these two values unequal
            condition = (pre != labels).float()

            # Filter out images that get either correct predictions or non-decreasing loss,
            # i.e., only images that are both misclassified and loss-decreasing are left
            mask = condition * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(self.steps // 10, 1) == 0:

                succeed = net(best_adv_images).argmax(dim=1) != labels
                bound = torch.linalg.norm((best_adv_images.detach() - images.detach()).flatten(), ord=2)

                if cost.item() > prev_cost:
                    return succeed, bound, best_adv_images.detach()

                prev_cost = cost.item()

        succeed = net(best_adv_images).argmax(dim=1) != labels
        bound = torch.linalg.norm((best_adv_images.detach() - images.detach()).flatten(), ord=2)
        return succeed, bound, best_adv_images.detach()


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

           :param image: Image of size HxWx3
           :param ground truth label
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
            r_i =  (pert+1e-4) * w / np.linalg.norm(w)
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
