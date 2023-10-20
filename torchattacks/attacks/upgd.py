import numpy as np

import torch
import torch.nn as nn

from ..attack import Attack


class UPGD(Attack):
    r"""
    Ultimate PGD that supports various options of gradient-based adversarial attacks.

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: False)
        loss (str): loss function. ['ce', 'margin', 'dlr'] (Default: 'ce')
        decay (float): momentum factor. (Default: 1.0)
        eot_iter (int) : number of models to estimate the mean gradient. (Default: 1)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.UPGD(model, eps=8/255, alpha=2/255, steps=10, random_start=False)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        model,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
        random_start=False,
        loss="ce",
        decay=1.0,
        eot_iter=1,
    ):
        super().__init__("UPGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.loss = loss
        self.decay = decay
        self.eot_iter = eot_iter
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            grad = torch.zeros_like(adv_images)
            adv_images.requires_grad = True

            for j in range(self.eot_iter):
                # Calculate loss
                if self.targeted:
                    cost = self.get_loss(adv_images, labels, target_labels)
                else:
                    cost = self.get_loss(adv_images, labels)

                grad += (
                    torch.autograd.grad(
                        cost, adv_images, retain_graph=False, create_graph=False
                    )[0]
                    / self.eot_iter
                )

            # Update adversarial images
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    def get_loss(self, images, labels, target_labels=None):
        if isinstance(self.loss, str):
            if self.loss == "ce":
                return self.ce_loss(images, labels, target_labels)
            elif self.loss == "dlr":
                return self.dlr_loss(images, labels, target_labels)
            elif self.loss == "margin":
                return self.margin_loss(images, labels, target_labels)
            else:
                raise ValueError(self.loss + " is not valid.")
        else:
            return self.loss(images, labels, target_labels)

    def ce_loss(self, images, labels, target_labels):
        loss = nn.CrossEntropyLoss()
        outputs = self.get_logits(images)

        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)
        return cost

    def dlr_loss(self, images, labels, target_labels):
        outputs = self.get_logits(images)
        outputs_sorted, ind_sorted = outputs.sort(dim=1)
        if self.targeted:
            cost = -(
                outputs[np.arange(outputs.shape[0]), labels]
                - outputs[np.arange(outputs.shape[0]), target_labels]
            ) / (
                outputs_sorted[:, -1]
                - 0.5 * outputs_sorted[:, -3]
                - 0.5 * outputs_sorted[:, -4]
                + 1e-12
            )
        else:
            ind = (ind_sorted[:, -1] == labels).float()
            cost = -(
                outputs[np.arange(outputs.shape[0]), labels]
                - outputs_sorted[:, -2] * ind
                - outputs_sorted[:, -1] * (1.0 - ind)
            ) / (outputs_sorted[:, -1] - outputs_sorted[:, -3] + 1e-12)
        return cost.sum()

    # f-function in the paper
    def margin_loss(self, images, labels, target_labels):
        outputs = self.get_logits(images)
        if self.targeted:
            # one_hot_labels = torch.eye(len(outputs[0]))[target_labels].to(self.device)
            one_hot_labels = torch.eye(outputs.shape[1]).to(self.device)[
                target_labels
            ]  # nopep8
            i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.bool())
            cost = -torch.clamp((i - j), min=0)  # -self.kappa=0
        else:
            one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)
            i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.bool())
            cost = -torch.clamp((j - i), min=0)  # -self.kappa=0
        return cost.sum()
