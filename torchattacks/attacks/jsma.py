import torch
import torch.nn as nn

from ..attack import Attack


class JSMA(Attack):
    r"""
    Jacobian Saliency Map Attack in the paper 'The Limitations of Deep Learning in Adversarial Settings'
    [https://arxiv.org/abs/1511.07528v1]

    This includes Algorithm 1 and 3 in v1

    Code is from
    [https://github.com/BorealisAI/advertorch/blob/master/advertorch/attacks/jsma.py]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        num_classes: number of clasess.
        gamma: highest percentage of pixels can be modified
        theta: perturb length, range is either [theta, 0], [0, theta]

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.JSMA(model, num_classes=10, gamma=1.0, theta=1.0)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, num_classes=10, gamma=1.0, theta=1.0):
        super().__init__("JSMA", model)
        self.num_classes = num_classes
        self.gamma = gamma
        self.theta = theta
        self.supported_mode = ['default']

    def jacobian(self, model, x, output_class):
        r"""
        Compute the output_class'th row of a Jacobian matrix. In other words,
        compute the gradient wrt to the output_class.
        Return output_class'th row of the Jacobian matrix wrt x.

        Arguments:
            model: forward pass function.
            x: input tensor.
            output_class: the output class we want to compute the gradients.
        """
        xvar = x.detach().clone().requires_grad_()
        scores = model(xvar)

        # compute gradients for the class output_class wrt the input x
        # using backpropagation
        torch.sum(scores[:, output_class]).backward()

        return xvar.grad

    def compute_forward_derivative(self, adv_images, labels):
        jacobians = torch.stack([self.jacobian(
            self.model, adv_images, adv_labels) for adv_labels in range(self.num_classes)])
        grads = jacobians.view((jacobians.shape[0], jacobians.shape[1], -1))
        grads_target = grads[labels, range(len(labels)), :]
        grads_other = grads.sum(dim=0) - grads_target
        return grads_target, grads_other

    def sum_pair(self, grads, dim_x):
        return grads.view(-1, dim_x, 1) + grads.view(-1, 1, dim_x)

    def and_pair(self, cond, dim_x):
        return cond.view(-1, dim_x, 1) & cond.view(-1, 1, dim_x)

    def saliency_map(self, search_space, grads_target, grads_other):
        dim_x = search_space.shape[1]
        # alpha in Algorithm 3 line 2
        gradsum_target = self.sum_pair(grads_target, dim_x)
        # alpha in Algorithm 3 line 3
        gradsum_other = self.sum_pair(grads_other, dim_x)

        if self.theta > 0:
            scores_mask = (torch.gt(gradsum_target, 0) &
                           torch.lt(gradsum_other, 0))
        else:
            scores_mask = (torch.lt(gradsum_target, 0) &
                           torch.gt(gradsum_other, 0))

        scores_mask &= self.and_pair(search_space.ne(0), dim_x)
        scores_mask[:, range(dim_x), range(dim_x)] = 0

        valid = torch.any(scores_mask.view(-1, dim_x * dim_x), dim=1)

        scores = scores_mask.float() * (-gradsum_target * gradsum_other)
        best = torch.max(scores.view(-1, dim_x * dim_x), 1)[1]
        p1 = torch.remainder(best, dim_x)
        p2 = (best / dim_x).long()
        return p1, p2, valid

    def modify_adv_images(self, adv_images, batch_size, cond, p1, p2):
        ori_shape = adv_images.shape
        adv_images = adv_images.view(batch_size, -1)
        for idx in range(batch_size):
            if cond[idx] != 0:
                adv_images[idx, p1[idx]] += self.theta
                adv_images[idx, p2[idx]] += self.theta
        adv_images = torch.clamp(adv_images, min=0, max=1)
        adv_images = adv_images.view(ori_shape)
        return adv_images

    def update_search_space(self, search_space, p1, p2, cond):
        for idx in range(len(cond)):
            if cond[idx] != 0:
                search_space[idx, p1[idx]] -= 1
                search_space[idx, p2[idx]] -= 1

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        self._check_inputs(images)

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = images
        batch_size = images.shape[0]
        dim_x = int(torch.prod(torch.tensor(images.shape[1:])))
        max_iters = int(dim_x * self.gamma / 2)
        search_space = images.new_ones(batch_size, dim_x, dtype=int)
        current_step = 0
        adv_pred = torch.argmax(self.get_logits(adv_images), 1)

        # Algorithm 1
        while (torch.any(labels != adv_pred) and current_step < max_iters):
            grads_target, grads_other = self.compute_forward_derivative(
                adv_images, labels)

            # Algorithm 3
            p1, p2, valid = self.saliency_map(
                search_space, grads_target, grads_other, labels)
            cond = (labels != adv_pred) & valid
            self.update_search_space(search_space, p1, p2, cond)

            adv_images = self.modify_adv_images(
                adv_images, batch_size, cond, p1, p2)
            adv_pred = torch.argmax(self.get_logits(adv_images), 1)

            current_step += 1

        adv_images = torch.clamp(adv_images, min=0, max=1)
        return adv_images
