import torch
import torch.nn as nn

from ..attack import Attack

class PGD(Attack):
    """
    PGD attack in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Arguments:
        model (nn.Module): a model to attack.
        eps (float): epsilon in the paper. (DEFALUT : 0.3)
        alpha (float): alpha in the paper. (DEFALUT : 2/255)
        iters (int): max iterations. (DEFALUT : 40)
        
    """
    def __init__(self, model, eps=0.3, alpha=2/255, iters=40):
        super(PGD, self).__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
    
    def forward(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()

        ori_images = images.clone().detach()
        
        for i in range(self.iters) :    
            images.requires_grad = True
            outputs = self.model(images)
            cost = loss(outputs, labels).to(self.device)
            
            grad = torch.autograd.grad(cost, images, 
                                       retain_graph=False, create_graph=False)[0]

            adv_images = images + self.alpha*grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-self.eps, max=self.eps)
            images = torch.clamp(ori_images + eta, min=0, max=1).detach()

        adv_images = images
        
        return adv_images  