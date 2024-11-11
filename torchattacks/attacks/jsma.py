import torch
import numpy as np

from ..attack import Attack


class JSMA(Attack):
    r"""
    Jacobian Saliency Map Attack in the paper 'The Limitations of Deep Learning in Adversarial Settings'
    [https://arxiv.org/abs/1511.07528v1]

    Distance Measure : L0

    Arguments:
        model (nn.Module): model to attack.
        theta (float): the change made to pixels. (Default: 1.0)
        gamma (float): the maximum distortion. (Default: 0.1)
        increasing (bool): crafting perturbation by increasing or decreasing pixel intensities. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.JSMA(model, theta=1.0, gamma=0.1, increasing=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, theta=1.0, gamma=0.1, increasing=True):
        super().__init__("JSMA", model)
        self.theta = theta
        self.gamma = gamma
        self.increasing = increasing
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        # only predict one image
        class_num = self.get_logits(torch.unsqueeze(images[0], 0)).shape[1]

        if self.targeted:
            target_labels = self.get_target_label(images, labels)
        else:
            # Because the JSMA algorithm does not use any loss function,
            # it cannot perform untargeted attacks indeed
            # (we have no control over the convergence of the attack to a data point that is NOT equal to the original class),
            # so we make the default setting of the target label is right circular shift
            # to make attack work if user didn't set target label.
            target_labels = (labels + 1) % class_num

        adv_images = images
        dim_x = int(np.prod(images.shape[1:]))
        max_iter = int(dim_x * self.gamma / 2)
        search_space = torch.ones(images.shape[0], dim_x).to(self.device)
        adv_prediction = torch.argmax(self.get_logits(adv_images), 1)

        # Algorithm 2
        i = 0
        while torch.sum(adv_prediction != target_labels) != 0 and i < max_iter and torch.sum(search_space != 0) != 0:
            grads_target, grads_other = self.compute_forward_derivative(
                adv_images, target_labels, class_num)
            p1, p2 = self.saliency_map(
                search_space, grads_target, grads_other, target_labels)

            cond = (adv_prediction != target_labels)
            self.update_search_space(search_space, p1, p2, cond)
            adv_images = self.update_adv_images(adv_images, p1, p2, cond)
            adv_prediction = torch.argmax(self.get_logits(adv_images), 1)
            i += 1

        adv_images = torch.clamp(adv_images, min=0, max=1)
        return adv_images

    def update_adv_images(self, adv_images, p1, p2, cond):
        origin_shape = adv_images.shape
        adv_images = adv_images.view(adv_images.shape[0], -1)
        for idx in range(adv_images.shape[0]):
            if cond[idx]:
                if self.increasing:
                    # Section IV, A
                    adv_images[idx, p1[idx]] += self.theta
                    adv_images[idx, p2[idx]] += self.theta
                else:
                    # Section IV, B
                    adv_images[idx, p1[idx]] -= self.theta
                    adv_images[idx, p2[idx]] -= self.theta

        adv_images = torch.clamp(adv_images, min=0, max=1)
        adv_images = adv_images.view(origin_shape)
        return adv_images

    def update_search_space(self, search_space, p1, p2, cond):
        # Algorithm 2 line 10 and line 11
        p1_cond = torch.logical_or(p1 == 0, p1 == 1)
        p2_cond = torch.logical_or(p2 == 0, p2 == 1)

        # Early stop
        p1_cond = torch.logical_or(p1_cond, cond)
        p2_cond = torch.logical_or(p2_cond, cond)

        for ind in range(search_space.shape[0]):
            if p1_cond[ind]:
                search_space[ind, p1[ind]] = False
            if p2_cond[ind]:
                search_space[ind, p2[ind]] = False

    def jacobian(self, adv_images, class_num):
        tmp_images = adv_images.detach().clone()
        tmp_images.requires_grad = True
        jacobians = []
        output = self.get_logits(tmp_images)

        for n in range(class_num):
            if tmp_images.grad is not None:
                tmp_images.grad.zero_()
            torch.sum(output[:, n]).backward(retain_graph=True)
            grad = tmp_images.grad.detach().clone()
            jacobians.append(grad)

        jacobians = torch.stack(jacobians, 0)
        return jacobians

    def compute_forward_derivative(self, adv_images, target_labels, class_num):
        jacobians = self.jacobian(adv_images, class_num)
        grads = jacobians.view((jacobians.shape[0], jacobians.shape[1], -1))
        grads_target = grads[target_labels, range(len(target_labels)), :]
        grads_other = grads.sum(dim=0) - grads_target
        return grads_target, grads_other

    def sum_pair(self, grads, dim):
        # Eq 8 and Eq 9
        return grads.view(-1, dim, 1) + grads.view(-1, 1, dim)

    def and_pair(self, cond, dim):
        return cond.view(-1, dim, 1) & cond.view(-1, 1, dim)

    def saliency_map(self, search_space, grads_target, grads_other, y):
        dim = search_space.shape[1]
        # alpha in Algorithm 3 line 2
        gradsum_target = self.sum_pair(grads_target, dim)
        # beta in Algorithm 3 line 3
        gradsum_other = self.sum_pair(grads_other, dim)

        # Algorithm 3 line 4
        if self.increasing:
            scores_mask = torch.logical_and(
                gradsum_target > 0, gradsum_other < 0)
        else:
            scores_mask = torch.logical_and(
                gradsum_target < 0, gradsum_other > 0)

        search_space_mask = self.and_pair(search_space != 0, dim)
        scores_mask = torch.logical_and(scores_mask, search_space_mask)
        scores_mask[:, range(dim), range(dim)] = 0
        scores = scores_mask.float() * (-1 * gradsum_target * gradsum_other)
        best_indices = torch.argmax(scores.view(-1, dim * dim), 1)

        p1 = torch.remainder(best_indices, dim)
        p2 = ((best_indices - p1) / dim).to(torch.long)
        return p1, p2
