import torch
import torch.nn as nn

from ..attack import Attack

class IFGSM(Attack):
    """
    I-FGSM attack in the paper 'Adversarial Examples in the Physical World'
    [https://arxiv.org/abs/1607.02533]

    Arguments:
        model (nn.Module): a model to attack.
        eps (float): epsilon in the paper. (DEFALUT : 4/255)
        alpha (float): alpha in the paper. (DEFALUT : 1/255)
        iters (int): max iterations. (DEFALUT : 0)
    
    .. note:: With 0 iters, iters will be automatically decided with the formula in the paper.
    """
    def __init__(self, model, eps=4/255, alpha=1/255, iters=0):
        super(IFGSM, self).__init__("IFGSM", model)
        self.eps = eps
        self.alpha = alpha
        if iters == 0 :
            self.iters = int(min(eps*255 + 4, 1.25*eps*255))
        else :
            self.iters = iters
        
    def forward(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()
        
        for i in range(self.iters) :    
            images.requires_grad = True
            outputs = self.model(images)

            self.model.zero_grad()
            cost = loss(outputs, labels).to(self.device)
            cost.backward()

            adv_images = images + self.alpha*images.grad.sign()
            
            a = torch.clamp(images - self.eps, min=0)
            b = (adv_images>=a).float()*adv_images + (a>adv_images).float()*a
            c = (b > images+self.eps).float()*(images+self.eps) + (images+self.eps >= b).float()*b
            images = torch.clamp(c, max=1).detach_()

        adv_images = images

        return adv_images