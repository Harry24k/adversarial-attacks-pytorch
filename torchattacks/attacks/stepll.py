import torch
import torch.nn as nn

from ..attack import Attack

class StepLL(Attack):
    """
    iterative least-likely class attack in the paper 'Adversarial Examples in the Physical World'
    [https://arxiv.org/abs/1607.02533]

    Arguments:
        model (nn.Module): a model to attack.
        eps (float): epsilon in the paper. (DEFALUT : 4/255)
        alpha (float): alpha in the paper. (DEFALUT : 1/255)
        iters (int): max iterations. (DEFALUT : 0)
    
    NOTE:: If iters set to 0, iters will be automatically decided following the paper.
    """
    def __init__(self, model, eps=4/255, alpha=1/255, iters=0):
        super(StepLL, self).__init__("StepLL", model)
        self.eps = eps
        self.alpha = alpha
        if iters == 0 :
            self.iters = int(min(eps*255 + 4, 1.25*eps*255))
        else :
            self.iters = iters
        
    def forward(self, images, labels):
        images = images.to(self.device)
        
        outputs = self.model(images)
        _, labels = torch.min(outputs.data, 1)
        labels = labels.detach_()
        
        loss = nn.CrossEntropyLoss()
        
        for i in range(self.iters) :    
            images.requires_grad = True
            outputs = self.model(images)
            cost = loss(outputs, labels).to(self.device)
            
            grad = torch.autograd.grad(cost, images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = images - self.alpha*grad.sign()
            
            a = torch.clamp(images - self.eps, min=0)
            b = (adv_images>=a).float()*adv_images + (a>adv_images).float()*a
            c = (b > images+self.eps).float()*(images+self.eps) + (images+self.eps >= b).float()*b
            images = torch.clamp(c, max=1).detach()

        adv_images = images

        return adv_images