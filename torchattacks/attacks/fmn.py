import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..attack import Attack


class FMN(Attack):
    r"""
    FMN in the paper 'Fast Minimum-norm Adversarial Attacks through Adaptive Norm Constraints'
    [https://arxiv.org/abs/2102.12827]
    """

    def __init__(self,
                 model: nn.Module,
                 device = None
                 ):
        super().__init__('FMN', model, device)
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            labels = self.get_target_label(images, labels)

        adv_images = images.clone().detach()
        batch_size = len(images)

        return
