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

        self.model.zero_grad()
        cost = loss(outputs, labels).to(self.device)
        cost.backward()

        adv_images = images + self.eps*images.grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach_()

        return adv_images
    
    