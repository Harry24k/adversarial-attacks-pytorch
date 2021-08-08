import torch

from ..attack import Attack, clamp_methods


class GN(Attack):
    r"""
    Add Gaussian Noise.

    Arguments:
        model (nn.Module): model to attack.
        sigma (nn.Module): sigma (Default: 0.1).
        clamp_function (function): function to clamp the output image see clamp_methods for examples

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.GN(model)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, sigma=0.1, clamp_function=clamp_methods.clamp_0_1):
        super().__init__("GN", model)
        self.sigma = sigma
        self._supported_mode = ['default']
        self.clamp_function = clamp_function

    def forward(self, images, labels=None):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        adv_images = images + self.sigma*torch.randn_like(images)
        adv_images = self.clamp_function(images, adv_images).detach()

        return adv_images
