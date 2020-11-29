import torch
import torch.nn as nn

from ..attack import Attack


class MIFGSM(Attack):
    r"""
    MI-FGSM in the paper 'Boosting Adversarial Attacks with Momentum'
    [https://arxiv.org/abs/1710.06081]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFALUT: 8/255)
        decay (float): momentum factor. (DEFAULT: 1.0)
        steps (int): number of iterations. (DEFAULT: 5)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.MIFGSM(model, eps=8/255, steps=5, decay=1.0)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8/255, steps=5, decay=1.0):
        super(MIFGSM, self).__init__("MIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = self.eps / self.steps

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        labels = self._transform_label(images, labels)
        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).to(self.device)

        for i in range(self.steps):
            images.requires_grad = True
            outputs = self.model(images)

            cost = self._targeted*loss(outputs, labels).to(self.device)
            grad = torch.autograd.grad(cost, images, 
                                       retain_graph=False, create_graph=False)[0]
            grad_norm = torch.norm(grad, p=1)
            grad /= grad_norm
            grad += momentum*self.decay
            momentum = grad

            adv_images = images + self.alpha*grad.sign()

            a = torch.clamp(images - self.eps, min=0)
            b = (adv_images >= a).float()*adv_images + (a > adv_images).float()*a
            c = (b > images + self.eps).float() * (images + self.eps) + (
                images + self.eps >= b
            ).float() * b
            images = torch.clamp(c, max=1).detach()

        adv_images = torch.clamp(images, min=0, max=1).detach()

        return adv_images
