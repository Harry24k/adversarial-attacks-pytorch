import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy import stats as st

from ..attack import Attack


class TIFGSM(Attack):
    r"""
    TIFGSM in the paper 'Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks'
    [https://arxiv.org/abs/1904.02884]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        kernel_name (str): kernel name. (DEFAULT: gaussian)
        len_kernel (int): kernel length.  (DEFAULT: 15, which is the best according to the paper)
        nsig (int): radius of gaussian kernel. (DEFAULT: 3; see Section 3.2.2 in the paper for explanation)
        eps (float): maximum perturbation. (DEFAULT: 8/255)
        alpha (float): step size. (DEFAULT: 2/255)
        decay (float): momentum factor. (DEFAULT: 0.0)
        steps (int): number of iterations. (DEFAULT: 20)
        resize_rate (float): resize factor used in input diversity. (DEFAULT: 0.9)
        diversity_prob (float) : the probability of applying input diversity. (DEFAULT: 0.5)
        random_start (bool): using random initialization of delta. (DEFAULT: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        # TIFGSM
        >>> attack = torchattacks.TIFGSM(model, eps=8/255, alpha=2/255, steps=20, decay=0.0, resize_rate=0.9, diversity_prob=0.0, random_start=False)
        >>> adv_images = attack(images, labels)
        # M-TIFGSM
        >>> attack = torchattacks.TIFGSM(model, eps=8/255, alpha=2/255, steps=20, decay=1.0, resize_rate=0.9, diversity_prob=0.0, random_start=False)
        >>> adv_images = attack(images, labels)
        # TI-DI-FGSM
        >>> attack = torchattacks.TIFGSM(model, eps=8/255, alpha=2/255, steps=20, decay=0.0, resize_rate=0.9, diversity_prob=0.7, random_start=False)
        >>> adv_images = attack(images, labels)
        # M-TI-DI-FGSM
        >>> attack = torchattacks.TIFGSM(model, eps=8/255, alpha=2/255, steps=20, decay=1.0, resize_rate=0.9, diversity_prob=0.7, random_start=False)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, kernel_name='gaussian', len_kernel=15, nsig=3, eps=8/255, alpha=2/255, steps=20, decay=0.0, resize_rate=0.9, diversity_prob=0.5, random_start=False):
        super(TIFGSM, self).__init__("TIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig

        self.stacked_kernel = torch.from_numpy(self.kernel_generation())

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)
        
        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]
            
        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x


    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels = self._transform_label(images, labels)
        
        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).detach().to(self.device)
        stacked_kernel = self.stacked_kernel.to(self.device)

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        
        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(self.input_diversity(adv_images))

            cost = self._targeted*loss(outputs, labels)
            
            grad = torch.autograd.grad(cost, adv_images, 
                                       retain_graph=False, create_graph=False)[0]
            # depth wise conv2d
            grad = F.conv2d(grad, stacked_kernel, stride=1, padding=int((self.len_kernel-1)/2), groups=3)
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() - self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel
