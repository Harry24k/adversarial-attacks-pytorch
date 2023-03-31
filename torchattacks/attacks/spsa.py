import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from ..attack import Attack


class MarginalLoss(_Loss):
    def forward(self, logits, targets):
        assert logits.shape[-1] >= 2
        top_logits, top_classes = torch.topk(logits, 2, dim=-1)
        target_logits = logits[torch.arange(logits.shape[0]), targets]
        max_nontarget_logits = torch.where(
            top_classes[..., 0] == targets,
            top_logits[..., 1],
            top_logits[..., 0],
        )

        loss = max_nontarget_logits - target_logits
        if self.reduction == "none":
            pass
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        else:
            raise ValueError("unknown reduction: '%s'" % (self.recution,))

        return loss


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
        delta (float): scaling parameter of SPSA.
        lr (float): the learning rate of the `Adam` optimizer.
        nb_iter (int): number of iterations of the attack.
        nb_sample (int): number of samples for SPSA gradient approximation.
        max_batch_size (int): maximum batch size to be evaluated at once.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.SPSA(model, eps=0.3)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=0.3, delta=0.01, lr=0.01, nb_iter=1, nb_sample=128, max_batch_size=64):
        super().__init__("SPSA", model)
        self.eps = eps
        self.delta = delta
        self.lr = lr
        self.nb_iter = nb_iter
        self.nb_sample = nb_sample
        self.max_batch_size = max_batch_size
        self.loss_fn = MarginalLoss(reduction="none")
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            def loss_fn(*args):
                return self.loss_fn(*args)
        else:
            def loss_fn(*args):
                return -self.loss_fn(*args)

        adv_images = self.spsa_perturb(loss_fn, images, labels)
        return adv_images

    def _batch_clamp_tensor_by_vector(self, vector, batch_tensor):
        return torch.min(torch.max(batch_tensor.transpose(0, -1), -vector), vector).transpose(0, -1).contiguous()

    def clamp(self, input, min=None, max=None):
        ndim = input.ndimension()
        if min is None:
            pass
        elif isinstance(min, (float, int)):
            input = torch.clamp(input, min=min)
        elif isinstance(min, torch.Tensor):
            if min.ndimension() == ndim - 1 and min.shape == input.shape[1:]:
                input = torch.max(input, min.view(1, *min.shape))
            else:
                assert min.shape == input.shape
                input = torch.max(input, min)
        else:
            raise ValueError("min can only be None | float | torch.Tensor")

        if max is None:
            pass
        elif isinstance(max, (float, int)):
            input = torch.clamp(input, max=max)
        elif isinstance(max, torch.Tensor):
            if max.ndimension() == ndim - 1 and max.shape == input.shape[1:]:
                input = torch.min(input, max.view(1, *max.shape))
            else:
                assert max.shape == input.shape
                input = torch.min(input, max)
        else:
            raise ValueError("max can only be None | float | torch.Tensor")
        return input

    def batch_clamp(self, float_or_vector, tensor):
        if isinstance(float_or_vector, torch.Tensor):
            assert len(float_or_vector) == len(tensor)
            tensor = self._batch_clamp_tensor_by_vector(
                float_or_vector, tensor)
            return tensor
        elif isinstance(float_or_vector, float):
            tensor = self.clamp(tensor, -float_or_vector, float_or_vector)
        else:
            raise TypeError("Value has to be float or torch.Tensor")
        return tensor

    def linf_clamp_(self, dx, x, eps, clip_min, clip_max):
        """Clamps perturbation `dx` to fit L_inf norm and image bounds.

        Limit the L_inf norm of `dx` to be <= `eps`, and the bounds of `x + dx`
        to be in `[clip_min, clip_max]`.

        Return: the clamped perturbation `dx`.
        """

        dx_clamped = self.batch_clamp(eps, dx)
        x_adv = self.clamp(x + dx_clamped, clip_min, clip_max)
        # `dx` is changed *inplace* so the optimizer will keep
        # tracking it. the simplest mechanism for inplace was
        # adding the difference between the new value `x_adv - x`
        # and the old value `dx`.
        dx += x_adv - x - dx
        return dx

    def _get_batch_sizes(self, n, max_batch_size):
        batches = [max_batch_size for _ in range(n // max_batch_size)]
        if n % max_batch_size > 0:
            batches.append(n % max_batch_size)
        return batches

    @torch.no_grad()
    def spsa_grad(self, loss_fn, images, labels, delta, nb_sample, max_batch_size):
        """Uses SPSA method to apprixmate gradient w.r.t `x`.

        Use the SPSA method to approximate the gradient of `loss_fn(predict(x), y)`
        with respect to `x`, based on the nonce `v`.

        Return the approximated gradient of `loss_fn(predict(x), y)` with respect to `x`.
        """

        grad = torch.zeros_like(images)
        images = images.unsqueeze(0)
        labels = labels.unsqueeze(0)

        def f(xvar, yvar):
            return loss_fn(self.get_logits(xvar), yvar)
        images = images.expand(max_batch_size, *images.shape[1:]).contiguous()
        labels = labels.expand(max_batch_size, *labels.shape[1:]).contiguous()
        v = torch.empty_like(images[:, :1, ...])

        for batch_size in self._get_batch_sizes(nb_sample, max_batch_size):
            x_ = images[:batch_size]
            y_ = labels[:batch_size]
            vb = v[:batch_size]
            vb = vb.bernoulli_().mul_(2.0).sub_(1.0)
            v_ = vb.expand_as(x_).contiguous()
            x_shape = x_.shape
            x_ = x_.view(-1, *images.shape[2:])
            y_ = y_.view(-1, *labels.shape[2:])
            v_ = v_.view(-1, *v.shape[2:])
            df = f(x_+delta*v_, y_) - f(x_-delta*v_, y_)
            df = df.view(-1, *[1 for _ in v_.shape[1:]])
            grad_ = df / (2.*delta*v_)
            grad_ = grad_.view(x_shape)
            grad_ = grad_.sum(dim=0, keepdim=False)
            grad += grad_

        grad /= nb_sample
        return grad

    def spsa_perturb(self, loss_fn, x, y):
        dx = torch.zeros_like(x)
        dx.grad = torch.zeros_like(dx)
        optimizer = torch.optim.Adam([dx], lr=self.lr)
        for _ in range(self.nb_iter):
            optimizer.zero_grad()
            dx.grad = self.spsa_grad(
                loss_fn, x + dx, y, self.delta, self.nb_sample, self.max_batch_size)
            optimizer.step()
            dx = self.linf_clamp_(dx, x, self.eps, 0, 1)

        x_adv = x + dx
        return x_adv
