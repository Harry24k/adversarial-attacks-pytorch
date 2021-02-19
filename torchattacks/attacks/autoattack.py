import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attack import Attack
from .multiattack import MultiAttack
from .apgd import APGD
from .apgdt import APGDT
from .fab import FAB
from .square import Square


class AutoAttack(Attack):
    r"""
    AutoAttack in the paper 'Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks'
    [https://arxiv.org/abs/2003.01690]
    [https://github.com/fra31/auto-attack]
    
    Distance Measure : Linf, L2

    Arguments:
        model (nn.Module): model to attack.
        norm (str) : Lp-norm to minimize. ('Linf', 'L2' supported, DEFALUT: 'Linf')
        eps (float): maximum perturbation. (DEFALUT: 0.3)
        version (bool): version. (DEFAULT: 'standard')
        n_classes (int): number of classes. (DEFAULT: 10)
        seed (int): random seed for the starting point. (DEFAULT: 0)
        verbose (bool): print progress. (DEFAULT: False)
        
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.AutoAttack(model, norm='Linf', eps=.3, version='standard', n_classes=10, seed=None, verbose=False)
        >>> adv_images = attack(images, labels)
        
    """
    def __init__(self, model, norm='Linf', eps=.3, version='standard', n_classes=10, seed=None, verbose=False):
        super(AutoAttack, self).__init__("AutoAttack", model)
        self.eps = eps
        self.norm = norm
        self.seed = seed
        self.verbose = verbose
        self.n_classes = n_classes
        self._attack_mode = 'only_default'
        
        if version == 'standard':
            self.autoattack = MultiAttack([
                APGD(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, loss='ce', n_restarts=1),
                APGDT(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, n_classes=n_classes, n_restarts=1),
                FAB(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, n_classes=n_classes, n_restarts=1),
                Square(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, n_queries=5000, n_restarts=1),
            ])
        
        elif version == 'plus':
            self.autoattack = MultiAttack([
                APGD(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, loss='ce', n_restarts=5),
                APGD(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, loss='dlr', n_restarts=5),
                FAB(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, n_classes=n_classes, n_restarts=5),
                Square(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, n_queries=5000, n_restarts=1),
                APGDT(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, n_classes=n_classes, n_restarts=1),
                FAB(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, targeted=True, n_classes=n_classes, n_restarts=1),
            ])

        elif version == 'rand':
            self.autoattack = MultiAttack([
                APGD(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, loss='ce', eot_iter=20, n_restarts=1),
                APGD(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, loss='dlr', eot_iter=20, n_restarts=1),
            ])
            
        else:
            raise ValueError("Not valid version. ['standard', 'plus', 'rand']")

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self.autoattack(images, labels)

        return adv_images
    
    def get_seed(self):
        return time.time() if self.seed is None else self.seed