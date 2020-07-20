import torch
import torch.nn as nn

from ..attack import Attack


class RFGSM(Attack):
    r"""
    'Ensemble Adversarial Training : Attacks and Defences'
    [https://arxiv.org/abs/1705.07204]

    RFGSM = Random Noise Start + FGSM

    Arguments:
        model (nn.Module): model to attack.
        eps (float): strength of the attack or maximum perturbation. (DEFALUT : 16/255)
        alpha (float): step size. (DEFALUT : 8/255)
        steps (int): number of steps. (DEFALUT : 1)
    
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.RFGSM(model, eps=16/255, alpha=8/255, steps=1)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, eps=16/255, alpha=8/255, steps=1):
        super(RFGSM, self).__init__("RFGSM", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach() + self.alpha*torch.randn_like(images).sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            cost = loss(outputs, labels).to(self.device)

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images + (self.eps-self.alpha)*grad.sign()
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
