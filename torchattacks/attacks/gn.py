import torch

from ..attack import Attack


class GN(Attack):
    r"""
    Add Gaussian Noise.

    Arguments:
        model (nn.Module): model to attack.
        std (nn.Module): standard deviation (Default: 0.1).

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.GN(model)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, std=0.1):
        super().__init__("GN", model)
        self.std = std
        self.supported_mode = ['default']

    def forward(self, images, labels=None):
        r"""
        Overridden.
        """
        if torch.max(images) > 1 or torch.min(images) < 0:
            print('Input must have a range [0, 1] (max: {}, min: {})'.format(torch.max(images), torch.min(images)))
            return torch.zeros(images.shape)

        images = images.clone().detach().to(self.device)
        adv_images = images + self.std*torch.randn_like(images)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
