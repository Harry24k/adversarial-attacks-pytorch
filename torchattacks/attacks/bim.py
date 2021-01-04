import torch
import torch.nn as nn

from ..attack import Attack


class BIM(Attack):
    r"""
    BIM or iterative-FGSM in the paper 'Adversarial Examples in the Physical World'
    [https://arxiv.org/abs/1607.02533]

    Distance Measure : Linf
    
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFALUT: 4/255)
        alpha (float): step size. (DEFALUT: 1/255)
        steps (int): number of steps. (DEFALUT: 0)
    
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
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels = self._transform_label(images, labels)
        
        loss = nn.CrossEntropyLoss()
        
        ori_images = images.clone().detach()

        for i in range(self.steps):
            images.requires_grad = True
            outputs = self.model(images)
            cost = self._targeted*loss(outputs, labels)

            grad = torch.autograd.grad(cost, images,
                                       retain_graph=False,
                                       create_graph=False)[0]

            adv_images = images - self.alpha*grad.sign()
            # a = max(ori_images-eps, 0)
            a = torch.clamp(ori_images - self.eps, min=0)
            # b = max(adv_images, a) = max(adv_images, ori_images-eps, 0)
            b = (adv_images >= a).float()*adv_images \
                + (adv_images < a).float()*a 
            # c = min(ori_images+eps, b) = min(ori_images+eps, max(adv_images, ori_images-eps, 0))
            c = (b > ori_images+self.eps).float()*(ori_images+self.eps) \
                + (b <= ori_images + self.eps).float()*b 
            # images = max(1, c) = min(1, ori_images+eps, max(adv_images, ori_images-eps, 0))
            images = torch.clamp(c, max=1).detach()

        return images
