import torch
import torch.nn as nn

from ..attack import Attack


class PGD(Attack):
    r"""
    PGD(Linf) attack in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Arguments:
        model (nn.Module): model to attack.
        eps (float): strength of the attack or maximum perturbation. (DEFALUT : 0.3)
        alpha (float): step size. (DEFALUT : 2/255)
        steps (int): number of steps. (DEFALUT : 40)
        random_start (bool): using random initialization of delta. (DEFAULT : False)
        targeted (bool): using targeted attack with input labels as targeted labels. (DEFAULT : False)
        
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.PGD(model, eps = 4/255, alpha = 8/255, steps=40, random_start=False)
        >>> adv_images = attack(images, labels)
        
    """
    def __init__(self, model, eps=0.3, alpha=2/255, steps=40, random_start=False, targeted=False):
        super(PGD, self).__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.sign = 1
        if targeted:
            self.sign = -1

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()

        ori_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            images = images + torch.empty_like(images).uniform_(-self.eps, self.eps)
            images = torch.clamp(images, min=0, max=1)

        for i in range(self.steps):
            images.requires_grad = True
            outputs = self.model(images)

            cost = self.sign*loss(outputs, labels).to(self.device)

            grad = torch.autograd.grad(cost, images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = images + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - ori_images, min=-self.eps, max=self.eps)
            images = torch.clamp(ori_images + delta, min=0, max=1).detach()

        adv_images = images

        return adv_images
