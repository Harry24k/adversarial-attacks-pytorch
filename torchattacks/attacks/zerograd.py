import torch
import torch.nn as nn

from ..attack import Attack


class ZeroGrad(Attack):
    r"""
    ZeroGrad in the paper 'ZeroGrad : Mitigating and Explaining Catastrophic Overfitting in FGSM Adversarial Training'
    [https://arxiv.org/abs/2103.15476]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 16/255)
        qval (float): quantile which gradients would become zero in zerograd (Default: 0.35)
        steps (int): number of zerograd iterations. (Default: 1)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.ZeroGrad(model, eps=8/255, alpha=16/255, qval=0.35, steps=1, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, alpha=16 / 255, qval=0.35, steps=1, random_start=True):
        super().__init__("ZeroGrad", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.qval = qval
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Zero out the small values in the gradient
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]
            q_grad = torch.quantile(torch.abs(grad).view(grad.size(0), -1), self.q_val, dim=1)
            grad[torch.abs(grad) < q_grad.view(grad.size(0), 1, 1, 1)] = 0

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            grad = delta.grad.detach()

        return adv_images
