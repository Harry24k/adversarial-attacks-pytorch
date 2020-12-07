import torch
import torch.nn as nn

from ..attack import Attack

class GN(Attack):
    r"""
    Add Gaussian Noise.

    Arguments:
        model (nn.Module): model to attack.
        sigma (nn.Module): sigma (DEFAULT: 0.1).
    
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.GN(model)
        >>> adv_images = attack(images, labels)
        
    """
    def __init__(self, model, sigma=0.1):
        super(GN, self).__init__("GN", model)
        self.sigma = sigma
        self._attack_mode = 'only_default'

    def forward(self, images, labels=None):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        adv_images = images + self.sigma*torch.randn_like(images)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images