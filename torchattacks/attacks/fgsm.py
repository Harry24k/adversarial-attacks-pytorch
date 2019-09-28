import torch
import torch.nn as nn

from ..attack import Attack

class FGSM(Attack):
    """
    FGSM attack in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Arguments:
        model (nn.Module): a model to attack.
        eps (float): epsilon in the paper. (DEFALUT : 0.007)
    
    """
    def __init__(self, model, eps=0.007):
        super(FGSM, self).__init__("FGSM", model)
        self.eps = eps
    
    def forward(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()
        
        images.requires_grad = True
        outputs = self.model(images)
        cost = loss(outputs, labels).to(self.device)
        
        grad = torch.autograd.grad(cost, images, 
                                   retain_graph=False, create_graph=False)[0]

        adv_images = images + self.eps*grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
    
    