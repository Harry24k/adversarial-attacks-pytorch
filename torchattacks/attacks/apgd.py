import torch
import torch.nn as nn

from ..attack import Attack

class APGD(Attack):
    """
    Comment on "Adv-BNN: Improved Adversarial Defense through Robust Bayesian Neural Network"
    [https://arxiv.org/abs/1907.00895]

    Arguments:
        model (nn.Module): a model to attack.
        eps (float): epsilon in the paper. (DEFALUT : 0.3)
        alpha (float): alpha in the paper. (DEFALUT : 2/255)
        iters (int): max iterations. (DEFALUT : 40)
        sampling (int) : the number of models to sample. (DEFALUT : 100)
        
    """
    def __init__(self, model, eps=0.3, alpha=2/255, iters=40, sampling=10):
        super(APGD, self).__init__("APGD", model)
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
        self.sampling = sampling
    
    def forward(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()

        ori_images = images.clone().detach()
        
        for i in range(self.iters) :    
            
            grad = torch.zeros_like(images)
            images.requires_grad = True
            
            for j in range(self.sampling) :
                
                outputs = self.model(images)
                cost = loss(outputs, labels).to(self.device)
                
                grad += torch.autograd.grad(cost, images, 
                                            retain_graph=False, create_graph=False)[0]
            
            # grad.sign() is used instead of (grad/sampling).sign()       
            adv_images = images + self.alpha*grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-self.eps, max=self.eps)
            images = torch.clamp(ori_images + eta, min=0, max=1).detach()

        adv_images = images
        
        return adv_images