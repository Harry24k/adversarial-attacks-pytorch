import torch
import torch.nn.functional as F
import numpy as np

from ..attack import Attack


class PIFGSM(Attack):
    r"""
    PIFGSM in the paper 'Patch-wise Attack for Fooling Deep Neural Network'
    [https://arxiv.org/abs/2007.06765]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        max_epsilon (float): maximum size of adversarial perturbation. (Default: 16/255)
        num_iter_set (float): number of iterations. (Default: 10)
        momentum (float): momentum. (Default: 1.0)
        amplification (float): to amplifythe step size. (Default: 10.0)
        prob (float): probability of using diverse inputs. (Default: 0.7)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PIFGSM(model, eps=16/255, num_iter_set=10)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        model,
        max_epsilon=16 / 255,
        num_iter_set=10,
        momentum=1.0,
        amplification=10.0,
        prob=0.7,
    ):
        super().__init__("PIFGSM", model)
        self.max_epsilon = max_epsilon
        self.num_iter_set = num_iter_set
        self.momentum = momentum
        self.amplification = amplification
        self.prob = prob
        self.supported_mode = ["default"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        images_min = self.clip_by_tensor(images - self.max_epsilon, t_min=0, t_max=1)
        images_max = self.clip_by_tensor(images + self.max_epsilon, t_min=0, t_max=1)
        adv_images = self.graph(images, labels, images_min, images_max)

        return adv_images

    def clip_by_tensor(self, t, t_min, t_max):
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def project_noise(self, images, P_kern, padding_size):
        images = F.conv2d(
            images, P_kern, padding=(padding_size, padding_size), groups=3
        )
        return images

    def project_kern(self, kern_size):
        kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
        kern[kern_size // 2, kern_size // 2] = 0.0
        kern = kern.astype(np.float32)
        stack_kern = np.stack([kern, kern, kern])
        stack_kern = np.expand_dims(stack_kern, 1)
        stack_kern = torch.tensor(stack_kern).to(self.device)
        return stack_kern, kern_size // 2

    def graph(self, images, labels, images_min, images_max):
        eps = self.max_epsilon
        num_iter = self.num_iter_set
        alpha = eps / num_iter
        alpha_beta = alpha * self.amplification
        gamma = alpha_beta
        P_kern, padding_size = self.project_kern(3)

        images.requires_grad = True
        amplification = 0.0
        for _ in range(num_iter):
            # zero_gradients(images)
            if images.grad is not None:
                images.grad.detach_()
                images.grad.zero_()

            output_v3 = self.get_logits(images)
            loss = F.cross_entropy(output_v3, labels)
            loss.backward()
            noise = images.grad.data

            amplification += alpha_beta * torch.sign(noise)
            cut_noise = torch.clamp(abs(amplification) - eps, 0, 10000.0) * torch.sign(
                amplification
            )
            projection = gamma * torch.sign(
                self.project_noise(cut_noise, P_kern, padding_size)
            )
            amplification += projection

            images = images + alpha_beta * torch.sign(noise) + projection
            images = self.clip_by_tensor(images, images_min, images_max)
            images = images.detach().requires_grad_(True)

        return images.detach()
