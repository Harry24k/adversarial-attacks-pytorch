import torch
import torch.nn as nn
import torch.optim as optim

from ..attack import Attack

class CW(Attack):
    """
    CW(L2) attack in the paper 'Towards Evaluating the Robustness of Neural Networks'
    [https://arxiv.org/abs/1608.04644]

    Arguments:
        model (nn.Module): a model to attack.
        targeted (bool):  (DEFALUT : False)
            True - change image closer to a given label
            False  - change image away from a right label
        c (float): c in the paper. (DEFALUT : 1e-4)
        kappa (float): kappa (also written as 'confidence') in the paper. (DEFALUT : 0)
        iters (int): max iterations. (DEFALUT : 1000)
        lr (float): learning rate of the 
        izer. (DEFALUT : 0.01)
        
    .. note:: There are serveral NOT IMPLEMENTED part of the paper/other codes.
    (1) Binary search method for c : It costs too many times.
    (2) Optimization on tanh space : Not in the paper, but in the other codes.
    (3) Choosing method best l2 adversaries : It will be implemented soon.
    
    .. warning:: With default c, you can't easily get adversarial images. Set higher c like 1.
    """
    def __init__(self, model, targeted=False, c=1e-4, kappa=0, iters=1000, lr=0.01):
        super(CW, self).__init__("CW", model)
        self.targeted = targeted
        self.c = c
        self.kappa = kappa
        self.iters = iters
        self.lr = lr
                
    def __call__(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # f-function in the paper
        def f(x) :

            outputs = self.model(x)
            one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

            i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.byte())

            # If targeted, optimize for making the other class most likely 
            if self.targeted :
                return torch.clamp(i-j, min=-self.kappa)

            # If untargeted, optimize for making the other class most likely 
            else :
                return torch.clamp(j-i, min=-self.kappa)
        
        w = torch.zeros_like(images).to(self.device)
        w.detach_()
        w.requires_grad=True
        
        optimizer = optim.Adam([w], lr=self.lr)
        prev = 1e10
                
        for step in range(self.iters) :    
            
            a = 1/2*(nn.Tanh()(w) + 1)

            loss1 = nn.MSELoss(reduction='sum')(a, images)
            loss2 = torch.sum(self.c*f(a))
            
            cost = loss1 + loss2

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Early Stop when loss does not converge.
            if step % (self.iters//10) == 0 :
                if cost > prev :
                    print('CW Attack is stopped due to CONVERGENCE....')
                    return a
                prev = cost
            
            print('- CW Attack Progress : %2.2f %%        ' %((step+1)/self.iters*100), end='\r')
            
        adv_images = (1/2*(nn.Tanh()(w) + 1)).detach_()

        return adv_images