import torch
import torch.nn as nn

from ..attack import Attack


class SPSA(Attack):
    r"""
    SPSA in the paper 'Adversarial Risk and the Dangers of Evaluating Against Weak Attacks'
    [https://arxiv.org/abs/1802.05666]
    Code is from
    [https://github.com/BorealisAI/advertorch/blob/master/advertorch/attacks/spsa.py]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        delta: scaling parameter of SPSA.
        lr: the learning rate of the `Adam` optimizer.
        nb_iter: number of iterations of the attack.
        nb_sample: number of samples for SPSA gradient approximation.
        max_batch_size: maximum batch size to be evaluated at once.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.SPSA(model, eps=8/255, delta=0.01)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=0.01, delta=0.01, lr=0.01, nb_iter=1, nb_sample=128, max_batch_size=64):
        super().__init__("SPSA", model)
        self.eps = eps
        self.delta = delta
        self.lr = lr
        self.nb_iter = nb_iter
        self.nb_sample = nb_sample
        self.max_batch_size = max_batch_size
        self.supported_mode = ['default', 'targeted']

    def spsa_grad(self, loss, x, y):
        r"""Use the SPSA method to approximate the gradient of `loss_fn(predict(x), y)`
        with respect to `x`, based on the nonce `v`.

        Return the approximated gradient of `loss(predict(x), y)` with respect to `x`.
        """
        grad = torch.zeros_like(x)
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

        def f(x, y):
            logits = self.get_logits(x)
            if self.targeted:
                return -loss(logits, y)
            else:
                return loss(logits, y)

        def get_batch_sizes(n, max_batch_size):
            batche_size = [max_batch_size] * (n // max_batch_size)
            if n % max_batch_size > 0:
                batche_size.append(n % max_batch_size)
            return batche_size

        x = x.expand(self.max_batch_size, *x.shape[1:]).contiguous()
        y = y.expand(self.max_batch_size, *y.shape[1:]).contiguous()
        v = torch.empty_like(x[:, :1, ...])

        for batch_size in get_batch_sizes(self.nb_sample, self.max_batch_size):
            x_ = x[:batch_size]
            y_ = y[:batch_size]
            vb = v[:batch_size]
            vb = vb.bernoulli_().mul_(2.0).sub_(1.0)
            v_ = vb.expand_as(x_).contiguous()
            x_shape = x_.shape
            x_ = x_.view(-1, *x.shape[2:])
            y_ = y_.view(-1, *y.shape[2:])
            v_ = v_.view(-1, *v.shape[2:])
            df = f(x_ + self.delta * v_, y_) - f(x_ - self.delta * v_, y_)
            df = df.view(-1, *[1 for _ in v_.shape[1:]])
            grad_ = df / (2. * self.delta * v_)
            grad_ = grad_.view(x_shape)
            grad_ = grad_.sum(dim=0, keepdim=False)
            grad += grad_

        grad /= self.nb_sample
        return grad

    def linf_clamp(self, dx, x):
        """Clamps perturbation `dx` to fit L_inf norm and image bounds.

        Limit the L_inf norm of `dx` to be <= `eps`, and the bounds of `x + dx`
        to be in `[clip_min, clip_max]`.

        Return the clamped perturbation `dx`.
        """
        dx_clamped = torch.clamp(dx, min=-self.eps, max=self.eps)
        x_adv = torch.clamp(x+dx_clamped, min=0, max=1)
        # `dx` is changed *inplace* so the optimizer will keep
        # tracking it. the simplest mechanism for inplace was
        # adding the difference between the new value `x_adv - x`
        # and the old value `dx`.
        dx += x_adv - x - dx
        return dx

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        self._check_inputs(images)

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        images.requires_grad = True

        # Update adversarial images
        dx = torch.zeros_like(images)
        dx.grad = torch.zeros_like(dx)
        optimizer = torch.optim.Adam([dx], lr=self.lr)
        for _ in range(self.nb_iter):
            optimizer.zero_grad()
            if self.targeted:
                dx.grad = self.spsa_grad(loss, images+dx, target_labels)
            else:
                dx.grad = self.spsa_grad(loss, images+dx, labels)
            optimizer.step()
            dx = self.linf_clamp(dx, images)

        adv_images = images + dx
        return adv_images
