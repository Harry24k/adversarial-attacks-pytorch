import torch
import torch.nn as nn

from ..attack import Attack

class RFGSM(Attack):
    """
    'Ensemble Adversarial Training : Attacks and Defences'
    [https://arxiv.org/abs/1705.07204]

    RFGSM = FGSM + Random Noise Start

    Arguments:
        model (nn.Module): a model to attack.
        eps (float): epsilon in the paper. (DEFALUT : 16/255)
        alpha (float): alpha in the paper. (DEFALUT : 8/255)
        iters (int): max iterations. (DEFALUT : 1)
    
    """
    def __init__(self, model, eps=16/255, alpha=8/255, iters=1):
        super(RFGSM, self).__init__("RFGSM", model)
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
    
    def forward(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()

        images = images + self.alpha*torch.randn_like(images).sign()

        for i in range(self.iters) :
            images.requires_grad = True
            outputs = self.model(images)
            cost = loss(outputs, labels).to(self.device)

            grad = torch.autograd.grad(cost, images, 
                                       retain_graph=False, create_graph=False)[0]
                
            adv_images = images + (self.eps-self.alpha)*grad.sign()
            images = torch.clamp(adv_images, min=0, max=1).detach_()

        adv_images = images
        
        return adv_images