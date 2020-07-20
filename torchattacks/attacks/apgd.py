import torch
import torch.nn as nn

from ..attack import Attack


class APGD(Attack):
    r"""
    Comment on "Adv-BNN: Improved Adversarial Defense through Robust Bayesian Neural Network"
    [https://arxiv.org/abs/1907.00895]

    Arguments:
        model (nn.Module): model to attack.
        eps (float): strength of the attack or maximum perturbation. (DEFALUT : 0.3)
        alpha (float): step size. (DEFALUT : 2/255)
        steps (int): number of steps. (DEFALUT : 40)
        sampling (int) : the number of models to sample. (DEFALUT : 100)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.APGD(model, eps = 4/255, alpha = 8/255, steps=40, sampling=100)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=0.3, alpha=2/255, steps=40, sampling=10):
        super(APGD, self).__init__("APGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.sampling = sampling

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()

        ori_images = images.clone().detach()

        for i in range(self.steps):

            grad = torch.zeros_like(images)
            images.requires_grad = True

            for j in range(self.sampling):

                outputs = self.model(images)
                cost = loss(outputs, labels).to(self.device)

                grad += torch.autograd.grad(cost, images,
                                            retain_graph=False,
                                            create_graph=False)[0]

            # grad.sign() is used instead of (grad/sampling).sign()
            adv_images = images + self.alpha*grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-self.eps, max=self.eps)
            images = torch.clamp(ori_images + eta, min=0, max=1).detach()

        adv_images = images

        return adv_images
