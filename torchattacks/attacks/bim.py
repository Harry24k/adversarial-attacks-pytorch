import torch
import torch.nn as nn

from ..attack import Attack


class BIM(Attack):
    r"""
    BIM or iterative-FGSM in the paper 'Adversarial Examples in the Physical World'
    [https://arxiv.org/abs/1607.02533]

    Arguments:
        model (nn.Module): model to attack.
        eps (float): strength of the attack or maximum perturbation. (DEFALUT : 4/255)
        alpha (float): step size. (DEFALUT : 1/255)
        steps (int): number of steps. (DEFALUT : 0)
    
    .. note:: If steps set to 0, steps will be automatically decided following the paper.
    
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.BIM(model, eps=4/255, alpha=1/255, steps=0)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, eps=4/255, alpha=1/255, steps=0):
        super(BIM, self).__init__("BIM", model)
        self.eps = eps
        self.alpha = alpha
        if steps == 0:
            self.steps = int(min(eps*255 + 4, 1.25*eps*255))
        else:
            self.steps = steps

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()

        for i in range(self.steps):
            images.requires_grad = True
            outputs = self.model(images)
            cost = loss(outputs, labels).to(self.device)

            grad = torch.autograd.grad(cost, images,
                                       retain_graph=False,
                                       create_graph=False)[0]

            adv_images = images + self.alpha*grad.sign()

            a = torch.clamp(images - self.eps, min=0)
            b = (adv_images >= a).float()*adv_images + (a > adv_images).float()*a
            c = (b > images+self.eps).float()*(images+self.eps) + (images+self.eps >= b).float()*b
            images = torch.clamp(c, max=1).detach()

        adv_images = images

        return adv_images
