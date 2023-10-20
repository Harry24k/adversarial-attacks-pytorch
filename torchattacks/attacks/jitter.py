import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attack import Attack


class Jitter(Attack):
    r"""
    Jitter in the paper 'Exploring Misclassifications of Robust Neural Networks to Enhance Adversarial Attacks'
    [https://arxiv.org/abs/2105.10304]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.Jitter(model, eps=8/255, alpha=2/255, steps=10,
                 scale=10, std=0.1, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        model,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
        scale=10,
        std=0.1,
        random_start=True,
    ):
        super().__init__("Jitter", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.scale = scale
        self.std = std
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.MSELoss(reduction="none")

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            logits = self.get_logits(adv_images)

            _, pre = torch.max(logits, dim=1)
            wrong = pre != labels

            norm_z = torch.norm(logits, p=float("inf"), dim=1, keepdim=True)
            hat_z = nn.Softmax(dim=1)(self.scale * logits / norm_z)

            if self.std != 0:
                hat_z = hat_z + self.std * torch.randn_like(hat_z)

            # Calculate loss
            if self.targeted:
                target_Y = F.one_hot(
                    target_labels, num_classes=logits.shape[-1]
                ).float()
                cost = -loss(hat_z, target_Y).mean(dim=1)
            else:
                Y = F.one_hot(labels, num_classes=logits.shape[-1]).float()
                cost = loss(hat_z, Y).mean(dim=1)

            norm_r = torch.norm(
                (adv_images - images), p=float("inf"), dim=[1, 2, 3]
            )  # nopep8
            nonzero_r = norm_r != 0
            cost[wrong * nonzero_r] /= norm_r[wrong * nonzero_r]

            cost = cost.mean()

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(
                adv_images - images, min=-self.eps, max=self.eps
            )  # nopep8
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
