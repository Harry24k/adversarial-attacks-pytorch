import torch
import torch.nn as nn

from ..attack import Attack
class NIFGSM(Attack):
    r"""
    NI-FGSM in the paper 'NESTEROV ACCELERATED GRADIENT AND SCALEINVARIANCE FOR ADVERSARIAL ATTACKS'
    [https://arxiv.org/abs/1908.06281], Published as a conference paper at ICLR 2020

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 1.0)
        steps (int): number of iterations. (Default: 10)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.NIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, decay=1.0, loss_func=None):
        super().__init__("NIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.supported_mode = ['default', 'targeted']
        if loss_func is None:
            self.loss_func = nn.CrossEntropyLoss()
        else:
            self.loss_func = loss_func

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = self.loss_func

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            nes_images = adv_images + self.decay*self.alpha*momentum
            outputs = self.get_logits(nes_images)
            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            grad = self.decay*momentum + grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            momentum = grad
            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
