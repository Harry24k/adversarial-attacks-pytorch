import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attack import Attack

class TPGD(Attack):
    r"""
    PGD(Linf) based on KL-Divergence loss in the paper 'Theoretically Principled Trade-off between Robustness and Accuracy'
    [https://arxiv.org/abs/1901.08573]

    Arguments:
        model (nn.Module): model to attack.
        eps (float): strength of the attack or maximum perturbation. (DEFALUT : 8/255)
        alpha (float): alpha in the paper. (DEFALUT : 2/255)
        iters (int): step size. (DEFALUT : 7)
        
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.TPGD(model, eps=8/255, alpha=2/255, iters=7)
        >>> adv_images = attack(images)
        
    """
    def __init__(self, model, eps=8/255, alpha=2/255, iters=7):
        super(TPGD, self).__init__("TPGD", model)
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
        
    def forward(self, images, labels=None):
        r"""
        Overridden.
        """
        images = images.to(self.device)
        adv_images = images.clone().detach() \
        + 0.001*torch.randn_like(images).to(self.device).detach()
        
        loss = nn.KLDivLoss(size_average=False)
        
        for i in range(self.iters) :    
            adv_images.requires_grad = True
            logit_ori = self.model(images)
            logit_adv = self.model(adv_images)
            
            cost = loss(F.log_softmax(logit_adv, dim=1),
                        F.softmax(logit_ori, dim=1)).to(self.device)
            
            grad = torch.autograd.grad(cost, adv_images, 
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            eta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(adv_images + eta, min=0, max=1).detach()
        
        return adv_images  