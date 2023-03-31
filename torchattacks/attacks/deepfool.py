import torch
import torch.nn as nn

from ..attack import Attack


class DeepFool(Attack):
    r"""
    'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
    [https://arxiv.org/abs/1511.04599]

    Distance Measure : L2

    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (Default: 50)
        overshoot (float): parameter for enhancing the noise. (Default: 0.02)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, steps=50, overshoot=0.02):
        super().__init__("DeepFool", model)
        self.steps = steps
        self.overshoot = overshoot
        self.supported_mode = ['default']

    def forward(self, images, labels, return_target_labels=False):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = None
        target_labels = []
        for image, label in zip(images, labels):
            adv_image = torch.unsqueeze(image, 0)
            label = torch.squeeze(label, 0)
            for _ in range(self.steps):
                early_stop, pre, adv_image = self._forward_indiv(adv_image, label)  # nopep8
                if early_stop:
                    try:
                        adv_images = torch.cat((adv_images, adv_image), 0)
                    except Exception:
                        adv_images = adv_image
                    target_labels.append(pre)
                    break

        # Fix for sparsefool attack
        adv_images = adv_images.detach()
        if return_target_labels:
            target_labels = torch.tensor(target_labels)
            return adv_images, target_labels

        return adv_images

    def _forward_indiv(self, image, label):
        # Only one image
        image.requires_grad = True
        fs = torch.squeeze(self.get_logits(image))
        pre = torch.argmax(fs)
        if pre != label:
            return (True, pre, image)

        ws = self._construct_jacobian(fs, image)
        image = image.detach()

        f_0 = fs[label]
        w_0 = ws[label]

        # wrong_classes = [i for i in range(len(fs)) if i != label]
        f_k = fs
        w_k = ws
        f_prime = f_k - f_0
        w_prime = w_k - w_0

        value = torch.abs(f_prime) / torch.norm(nn.Flatten()(w_prime), p=2, dim=1)  # nopep8
        value[label] = float('inf')
        hat_L = torch.argmin(value)

        delta = (torch.abs(f_prime[hat_L])*w_prime[hat_L] / (torch.norm(w_prime[hat_L], p=2)**2))  # nopep8

        target_label = hat_L

        adv_image = image + (1+self.overshoot)*delta
        adv_image = torch.clamp(adv_image, min=0, max=1).detach()
        return (False, target_label, adv_image)

    def _construct_jacobian(self, y, x):
        # https://stackoverflow.com/questions/63096122/pytorch-is-it-possible-to-differentiate-a-matrix
        # torch.autograd.functional.jacobian is only for torch >= 1.5.1
        x_grads = []
        for idx, y_element in enumerate(y):
            if x.grad is not None:
                x.grad.zero_()
            y_element.backward(retain_graph=(False or idx+1 < len(y)))
            x_grads.append(x.grad.clone().detach())
        return torch.stack(x_grads).reshape(*y.shape, *x.shape)
