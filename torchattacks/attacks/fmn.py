import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..attack import Attack


class FMN(Attack):
    r"""
    FMN in the paper 'Fast Minimum-norm Adversarial Attacks through Adaptive Norm Constraints'
    [https://arxiv.org/abs/2102.12827]

    Distance Measure : L0, L1, L2, Linf

    Args:
        model (nn.Module): The model to be attacked.
        norm (float): The norm for distance measure. Defaults to float('inf').
        steps (int): The number of steps for the attack. Defaults to 10.
        alpha_init (float): The initial alpha for the attack. Defaults to 1.0.
        alpha_final (Optional[float]): The final alpha for the attack. Defaults to alpha_init / 100 if not provided.
        gamma_init (float): The initial gamma for the attack. Defaults to 0.05.
        gamma_final (float): The final gamma for the attack. Defaults to 0.001.
        starting_points (Optional[Tensor]): The starting points for the attack. Defaults to None.
        binary_search_steps (int): The number of binary search steps. Defaults to 10.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples:
        >>> attack = torchattacks.FMN(model, norm=float('inf'), steps=10)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self,
                 model: nn.Module,
                 norm: float = float('inf'),
                 steps: int = 10,
                 alpha_init: float = 1.0,
                 alpha_final: Optional[float] = None,
                 gamma_init: float = 0.05,
                 gamma_final: float = 0.001,
                 starting_points: Optional[Tensor] = None,
                 binary_search_steps: int = 10
                 ):
        super().__init__('FMN', model)
        self.norm = norm
        self.steps = steps
        self.alpha_init = alpha_init
        self.alpha_final = self.alpha_init / 100 if alpha_final is None else alpha_final
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.starting_points = starting_points
        self.binary_search_steps = binary_search_steps
        self._dual_projection_mid_points = {
            0: (None, self._l0_projection, self._l0_mid_points),
            1: (float('inf'), self._l1_projection, self._l1_mid_points),
            2: (2, self._l2_projection, self._l2_mid_points),
            float('inf'): (1, self._linf_projection, self._linf_mid_points),
        }

        self.supported_mode = ['default', 'targeted']

    def _simplex_projection(self, x, epsilon):
        """
        Simplex projection based on sorting.
        Parameters
        ----------
        x : Tensor
            Batch of vectors to project on the simplex.
        epsilon : float or Tensor
            Size of the simplex, default to 1 for the probability simplex.
        Returns
        -------
        projected_x : Tensor
            Batch of projected vectors on the simplex.
        """
        u = x.sort(dim=1, descending=True)[0]
        epsilon = epsilon.unsqueeze(1) if isinstance(epsilon, Tensor) else torch.tensor(epsilon, device=x.device)
        indices = torch.arange(x.size(1), device=x.device)
        cumsum = torch.cumsum(u, dim=1).sub_(epsilon).div_(indices + 1)
        k = (cumsum < u).long().mul_(indices).amax(dim=1, keepdim=True)
        tau = cumsum.gather(1, k)
        return (x - tau).clamp_(min=0)

    def _l1_ball_euclidean_projection(self, x, epsilon, inplace):
        """
        Compute Euclidean projection onto the L1 ball for a batch.

          min ||x - u||_2 s.t. ||u||_1 <= eps

        Inspired by the corresponding numpy version by Adrien Gaidon.
        Adapted from Tony Duan's implementation https://gist.github.com/tonyduan/1329998205d88c566588e57e3e2c0c55

        Parameters
        ----------
        x: Tensor
            Batch of tensors to project.
        epsilon: float or Tensor
            Radius of L1-ball to project onto. Can be a single value for all tensors in the batch or a batch of values.
        inplace : bool
            Can optionally do the operation in-place.

        Returns
        -------
        projected_x: Tensor
            Batch of projected tensors with the same shape as x.

        Notes
        -----
        The complexity of this algorithm is in O(dlogd) as it involves sorting x.

        References
        ----------
        [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions
            John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
            International Conference on Machine Learning (ICML 2008)
        """
        to_project = x.norm(p=1, dim=1) > epsilon
        if to_project.any():
            x_to_project = x[to_project]
            epsilon_ = epsilon[to_project] if isinstance(epsilon, Tensor) else torch.tensor([epsilon], device=x.device)
            if not inplace:
                x = x.clone()
            simplex_proj = self._simplex_projection(x_to_project.abs(), epsilon=epsilon_)
            x[to_project] = simplex_proj.copysign_(x_to_project)
            return x
        else:
            return x

    def _l0_projection(self, delta, epsilon):
        """In-place l0 projection"""
        delta = delta.flatten(1)
        delta_abs = delta.abs()
        sorted_indices = delta_abs.argsort(dim=1, descending=True).gather(1, (epsilon.long().unsqueeze(1) - 1).clamp_(
            min=0))
        thresholds = delta_abs.gather(1, sorted_indices)
        delta.mul_(delta_abs >= thresholds)

    def _l1_projection(self, delta, epsilon):
        """In-place l1 projection"""
        self._l1_ball_euclidean_projection(x=delta.flatten(1), epsilon=epsilon, inplace=True)

    def _l2_projection(self, delta, epsilon):
        """In-place l2 projection"""
        delta = delta.flatten(1)
        l2_norms = delta.norm(p=2, dim=1, keepdim=True).clamp_(min=1e-12)
        delta.mul_(epsilon.unsqueeze(1) / l2_norms).clamp_(max=1)

    def _linf_projection(self, delta, epsilon):
        """In-place linf projection"""
        delta = delta.flatten(1)
        epsilon = epsilon.unsqueeze(1)
        torch.maximum(torch.minimum(delta, epsilon, out=delta), -epsilon, out=delta)

    def _l0_mid_points(self, x0, x1, epsilon):
        n_features = x0[0].numel()
        delta = x1 - x0
        self._l0_projection_(delta=delta, epsilon=n_features * epsilon)
        return delta

    def _l1_mid_points(self, x0, x1, epsilon):
        threshold = (1 - epsilon).unsqueeze(1)
        delta = (x1 - x0).flatten(1)
        delta_abs = delta.abs()
        mask = delta_abs > threshold
        mid_points = delta_abs.sub_(threshold).copysign_(delta)
        mid_points.mul_(mask)
        return x0 + mid_points

    def _l2_mid_points(self, x0, x1, epsilon):
        epsilon = epsilon.unsqueeze(1)
        return x0.flatten(1).mul(1 - epsilon).add_(epsilon * x1.flatten(1)).view_as(x0)

    def _linf_mid_points(self, x0, x1, epsilon):
        epsilon = epsilon.unsqueeze(1)
        delta = (x1 - x0).flatten(1)
        return x0 + torch.maximum(torch.minimum(delta, epsilon, out=delta), -epsilon, out=delta).view_as(x0)

    def _difference_of_logits(self, logits, labels, labels_infhot):
        if labels_infhot is None:
            labels_infhot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), float('inf'))

        class_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        other_logits = (logits - labels_infhot).amax(dim=1)
        return class_logits - other_logits

    def _boundary_search(self, images, labels):
        batch_size = len(images)
        _, _, mid_point = self._dual_projection_mid_points[self.norm]

        is_adv = self.model(self.starting_points).argmax(dim=1)
        if not is_adv.all():
            raise ValueError('Starting points are not all adversarial.')
        lower_bound = torch.zeros(batch_size, device=self.device)
        upper_bound = torch.ones(batch_size, device=self.device)
        for _ in range(self.binary_search_steps):
            epsilon = (lower_bound + upper_bound) / 2
            mid_points = mid_point(x0=images, x1=self.starting_points, epsilon=epsilon)
            pred_labels = self.model(mid_points).argmax(dim=1)
            is_adv = (pred_labels == labels) if self.targeted else (pred_labels != labels)
            lower_bound = torch.where(is_adv, lower_bound, epsilon)
            upper_bound = torch.where(is_adv, epsilon, upper_bound)

        delta = mid_point(x0=images, x1=self.starting_points, epsilon=epsilon) - images

        return epsilon, delta, is_adv

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            labels = self.get_target_label(images, labels)

        adv_images = images.clone().detach()

        batch_size = len(images)

        dual, projection, _ = self._dual_projection_mid_points[self.norm]
        batch_view = lambda tensor: tensor.view(batch_size, *[1] * (images.ndim - 1))

        delta = torch.zeros_like(images, device=self.device)
        is_adv = None

        if self.starting_points is not None:
            epsilon, delta, is_adv = self._boundary_search(images, labels)

        if self.norm == 0:
            epsilon = torch.ones(batch_size,
                                 device=self.device) if self.starting_points is None else delta.flatten(1).norm(p=0,
                                                                                                                dim=0)
        else:
            epsilon = torch.full((batch_size,), float('inf'), device=self.device)

        _worst_norm = torch.maximum(images, 1 - images).flatten(1).norm(p=self.norm, dim=1).detach()

        init_trackers = {
            'worst_norm': _worst_norm.to(self.device),
            'best_norm': _worst_norm.clone().to(self.device),
            'best_adv': adv_images,
            'adv_found': torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        }

        multiplier = 1 if self.targeted else -1
        delta.requires_grad_(True)

        optimizer = SGD([delta], lr=self.alpha_init)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.steps)

        for i in range(self.steps):
            optimizer.zero_grad()

            cosine = (1 + math.cos(math.pi * i / self.steps)) / 2
            gamma = self.gamma_final + (self.gamma_init - self.gamma_final) * cosine

            delta_norm = delta.data.flatten(1).norm(p=self.norm, dim=1)
            adv_images = images + delta
            adv_images = adv_images.to(self.device)

            logits = self.model(adv_images)
            pred_labels = logits.argmax(dim=1)

            if i == 0:
                labels_infhot = torch.zeros_like(logits).scatter_(
                    1,
                    labels.unsqueeze(1),
                    float('inf')
                )
                logit_diff_func = partial(
                    self._difference_of_logits,
                    labels=labels,
                    labels_infhot=labels_infhot
                )

            logit_diffs = logit_diff_func(logits=logits)
            loss = -(multiplier * logit_diffs)

            loss.sum().backward()

            delta_grad = delta.grad.data

            is_adv = (pred_labels == labels) if self.targeted else (pred_labels != labels)
            is_smaller = delta_norm < init_trackers['best_norm']
            is_both = is_adv & is_smaller
            init_trackers['adv_found'].logical_or_(is_adv)
            init_trackers['best_norm'] = torch.where(is_both, delta_norm, init_trackers['best_norm'])
            init_trackers['best_adv'] = torch.where(batch_view(is_both), adv_images.detach(),
                                                    init_trackers['best_adv'])

            if self.norm == 0:
                epsilon = torch.where(is_adv,
                                      torch.minimum(torch.minimum(epsilon - 1,
                                                                  (epsilon * (1 - gamma)).floor_()),
                                                    init_trackers['best_norm']),
                                      torch.maximum(epsilon + 1, (epsilon * (1 + gamma)).floor_()))
                epsilon.clamp_(min=0)
            else:
                distance_to_boundary = loss.detach().abs() / delta_grad.flatten(1).norm(p=dual, dim=1).clamp_(min=1e-12)
                epsilon = torch.where(is_adv,
                                      torch.minimum(epsilon * (1 - gamma), init_trackers['best_norm']),
                                      torch.where(init_trackers['adv_found'],
                                                  epsilon * (1 + gamma),
                                                  delta_norm + distance_to_boundary)
                                      )

            # clip epsilon
            epsilon = torch.minimum(epsilon, init_trackers['worst_norm'])

            # normalize gradient
            grad_l2_norms = delta_grad.flatten(1).norm(p=2, dim=1).clamp_(min=1e-12)
            delta_grad.div_(batch_view(grad_l2_norms))

            optimizer.step()

            # project in place
            projection(delta=delta.data, epsilon=epsilon)
            # clamp
            delta.data.add_(images).clamp_(min=0, max=1).sub_(images)

            scheduler.step()

        return init_trackers['best_adv']
