import torch
import torch.nn as nn

from ..attack import Attack
from ..clamp_methods import clamp_0_1


class RFGSM(Attack):
    r"""
    R+FGSM in the paper 'Ensemble Adversarial Training : Attacks and Defences'
    [https://arxiv.org/abs/1705.07204]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): strength of the attack or maximum perturbation. (Default: 16/255)
        alpha (float): step size. (Default: 8/255)
        steps (int): number of steps. (Default: 1)
        clamp_function (function): function to clamp the output image see clamp_methods for examples

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.RFGSM(model, eps=16/255, alpha=8/255, steps=1)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, eps=16/255, alpha=8/255, steps=1, clamp_function=clamp_0_1):
        super().__init__("RFGSM", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self._supported_mode = ['default', 'targeted']
        self.clamp_function = clamp_function

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images + self.alpha*torch.randn_like(images).sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = self.clamp_function(images, images + delta).detach()

        return adv_images
