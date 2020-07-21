import torch
import torch.nn as nn

from ..attack import Attack


class FFGSM(Attack):
    r"""
    'Fast is better than free: Revisiting adversarial training'
    [https://arxiv.org/abs/2001.03994]

    FFGSM = Random Noise Start + Large Alpha + FGSM

    Arguments:
        model (nn.Module): model to attack.
        eps (float): strength of the attack or maximum perturbation. (DEFALUT : 8/255)
        alpha (float): step size. (DEFALUT : 10/255)
    
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.FFGSM(model, eps=8/255, alpha=10/255)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, eps=8/255, alpha=10/255):
        super(FFGSM, self).__init__("FFGSM", model)
        self.eps = eps
        self.alpha = alpha

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()
        adv_images = adv_images + torch.randn_like(images).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        adv_images.requires_grad = True

        outputs = self.model(adv_images)
        cost = loss(outputs, labels).to(self.device)

        grad = torch.autograd.grad(cost, adv_images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = adv_images + self.alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
