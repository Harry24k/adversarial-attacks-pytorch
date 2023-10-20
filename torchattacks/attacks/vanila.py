from ..attack import Attack


class VANILA(Attack):
    r"""
    Vanila version of Attack.
    It just returns the input images.

    Arguments:
        model (nn.Module): model to attack.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.VANILA(model)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model):
        super().__init__("VANILA", model)
        self.supported_mode = ["default"]

    def forward(self, images, labels=None):
        r"""
        Overridden.
        """

        adv_images = images.clone().detach().to(self.device)

        return adv_images
