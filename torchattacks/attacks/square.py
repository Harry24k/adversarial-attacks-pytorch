from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time
import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attack import Attack

DEFAULT_EPS_DICT_BY_NORM = {'Linf': .3, 'L2': 1., 'L1': 5.0}

class Square(Attack):
    r"""
    Square Attack in the paper 'Square Attack: a query-efficient black-box adversarial attack via random search'
    [https://arxiv.org/abs/1912.00049]
    [https://github.com/fra31/auto-attack]
    
    Distance Measure : Linf, L2

    Arguments:
        model (nn.Module): model to attack.
        norm (str): Lp-norm of the attack. ('Linf', 'L2' supported, DEFAULT: 'Linf')
        eps (float): maximum perturbation. (DEFALUT: None)
        n_queries (int): max number of queries (each restart). (DEFALUT: 5000)
        n_restarts (int): number of random restarts. (DEFALUT: 1)
        p_init (float): parameter to control size of squares. (DEFALUT: 0.8)
        loss (str): loss function optimized ('margin', 'ce' supported, DEFALUT: 'margin')
        resc_schedule (bool): adapt schedule of p to n_queries (DEFAULT: True)
        seed (int): random seed for the starting point. (DEFAULT: 0)
        verbose (bool): print progress. (DEFAULT: False)
        targeted (bool): targeted. (DEFAULT: False)
        
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.Square(model, model, norm='Linf', n_queries=5000, n_restarts=1, eps=None, p_init=.8, seed=0, verbose=False, targeted=False, loss='margin', resc_schedule=True)
        >>> adv_images = attack(images, labels)
        
    """
    def __init__(self, model, norm='Linf', eps=None, n_queries=5000, n_restarts=1,
                 p_init=.8, loss='margin', resc_schedule=True,
                 seed=0, verbose=False, targeted=False):
        super(Square, self).__init__("Square", model)
        self.norm = norm
        self.n_queries = n_queries
        self.eps = eps
        self.p_init = p_init
        self.n_restarts = n_restarts
        self.seed = seed
        self.verbose = verbose
        self.targeted = targeted
        self.loss = loss
        self.rescale_schedule = resc_schedule        
        self._attack_mode = 'only_default'

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self.perturb(images, labels)

        return adv_images
    
    def margin_and_loss(self, x, y):
        """
        :param y:        correct labels if untargeted else target labels
        """

        logits = self.model(x)
        xent = F.cross_entropy(logits, y, reduction='none')
        u = torch.arange(x.shape[0])
        y_corr = logits[u, y].clone()
        logits[u, y] = -float('inf')
        y_others = logits.max(dim=-1)[0]

        if not self.targeted:
            if self.loss == 'ce':
                return y_corr - y_others, -1. * xent
            elif self.loss == 'margin':
                return y_corr - y_others, y_corr - y_others
        else:
            return y_others - y_corr, xent

    def init_hyperparam(self, x):
        assert self.norm in ['Linf', 'L2']
        assert not self.eps is None
        assert self.loss in ['ce', 'margin']

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()

    def random_target_classes(self, y_pred, n_classes):
        y = torch.zeros_like(y_pred)
        for counter in range(y_pred.shape[0]):
            l = list(range(n_classes))
            l.remove(y_pred[counter])
            t = self.random_int(0, len(l))
            y[counter] = l[t]

        return y.long().to(self.device)

    def check_shape(self, x):
        return x if len(x.shape) == (self.ndims + 1) else x.unsqueeze(0)

    def random_choice(self, shape):
        t = 2 * torch.rand(shape).to(self.device) - 1
        return torch.sign(t)

    def random_int(self, low=0, high=1, shape=[1]):
        t = low + (high - low) * torch.rand(shape).to(self.device)
        return t.long()

    def normalize(self, x):
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

    def lp_norm(self, x):
        if self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return t.view(-1, *([1] * self.ndims))

    def eta_rectangles(self, x, y):
        delta = torch.zeros([x, y]).to(self.device)
        x_c, y_c = x // 2 + 1, y // 2 + 1

        counter2 = [x_c - 1, y_c - 1]
        for counter in range(0, max(x_c, y_c)):
          delta[max(counter2[0], 0):min(counter2[0] + (2*counter + 1), x),
              max(0, counter2[1]):min(counter2[1] + (2*counter + 1), y)
              ] += 1.0/(torch.Tensor([counter + 1]).view(1, 1).to(
              self.device) ** 2)
          counter2[0] -= 1
          counter2[1] -= 1

        delta /= (delta ** 2).sum(dim=(0,1), keepdim=True).sqrt()
    
        return delta

    def eta(self, s):
        delta = torch.zeros([s, s]).to(self.device)
        delta[:s // 2] = self.eta_rectangles(s // 2, s)
        delta[s // 2:] = -1. * self.eta_rectangles(s - s // 2, s)
        delta /= (delta ** 2).sum(dim=(0, 1), keepdim=True).sqrt()
        if torch.rand([1]) > 0.5:
            delta = delta.permute([1, 0])

        return delta

    def p_selection(self, it):
        """ schedule to decrease the parameter p """

        if self.rescale_schedule:
            it = int(it / self.n_queries * 10000)

        if 10 < it <= 50:
            p = self.p_init / 2
        elif 50 < it <= 200:
            p = self.p_init / 4
        elif 200 < it <= 500:
            p = self.p_init / 8
        elif 500 < it <= 1000:
            p = self.p_init / 16
        elif 1000 < it <= 2000:
            p = self.p_init / 32
        elif 2000 < it <= 4000:
            p = self.p_init / 64
        elif 4000 < it <= 6000:
            p = self.p_init / 128
        elif 6000 < it <= 8000:
            p = self.p_init / 256
        elif 8000 < it:
            p = self.p_init / 512
        else:
            p = self.p_init

        return p

    def attack_single_run(self, x, y):
        with torch.no_grad():
            adv = x.clone()
            c, h, w = x.shape[1:]
            n_features = c * h * w
            n_ex_total = x.shape[0]
            
            if self.norm == 'Linf':
                x_best = torch.clamp(x + self.eps * self.random_choice(
                    [x.shape[0], c, 1, w]), 0., 1.)
                margin_min, loss_min = self.margin_and_loss(x_best, y)
                n_queries = torch.ones(x.shape[0]).to(self.device)
                s_init = int(math.sqrt(self.p_init * n_features / c))
                
                for i_iter in range(self.n_queries):
                    idx_to_fool = (margin_min > 0.0).nonzero().squeeze()
                    
                    x_curr = self.check_shape(x[idx_to_fool])
                    x_best_curr = self.check_shape(x_best[idx_to_fool])
                    y_curr = y[idx_to_fool]
                    if len(y_curr.shape) == 0:
                        y_curr = y_curr.unsqueeze(0)
                    margin_min_curr = margin_min[idx_to_fool]
                    loss_min_curr = loss_min[idx_to_fool]
                    
                    p = self.p_selection(i_iter)
                    s = max(int(round(math.sqrt(p * n_features / c))), 1)
                    vh = self.random_int(0, h - s)
                    vw = self.random_int(0, w - s)
                    new_deltas = torch.zeros([c, h, w]).to(self.device)
                    new_deltas[:, vh:vh + s, vw:vw + s
                        ] = 2. * self.eps * self.random_choice([c, 1, 1])
                    
                    x_new = x_best_curr + new_deltas
                    x_new = torch.min(torch.max(x_new, x_curr - self.eps),
                        x_curr + self.eps)
                    x_new = torch.clamp(x_new, 0., 1.)
                    x_new = self.check_shape(x_new)
                    
                    margin, loss = self.margin_and_loss(x_new, y_curr)

                    # update loss if new loss is better
                    idx_improved = (loss < loss_min_curr).float()

                    loss_min[idx_to_fool] = idx_improved * loss + (
                        1. - idx_improved) * loss_min_curr

                    # update margin and x_best if new loss is better
                    # or misclassification
                    idx_miscl = (margin <= 0.).float()
                    idx_improved = torch.max(idx_improved, idx_miscl)

                    margin_min[idx_to_fool] = idx_improved * margin + (
                        1. - idx_improved) * margin_min_curr
                    idx_improved = idx_improved.reshape([-1,
                        *[1]*len(x.shape[:-1])])
                    x_best[idx_to_fool] = idx_improved * x_new + (
                        1. - idx_improved) * x_best_curr
                    n_queries[idx_to_fool] += 1.

                    ind_succ = (margin_min <= 0.).nonzero().squeeze()
                    if self.verbose and ind_succ.numel() != 0:
                        print('{}'.format(i_iter + 1),
                            '- success rate={}/{} ({:.2%})'.format(
                            ind_succ.numel(), n_ex_total,
                            float(ind_succ.numel()) / n_ex_total),
                            '- avg # queries={:.1f}'.format(
                            n_queries[ind_succ].mean().item()),
                            '- med # queries={:.1f}'.format(
                            n_queries[ind_succ].median().item()),
                            '- loss={:.3f}'.format(loss_min.mean()))

                    if ind_succ.numel() == n_ex_total:
                        break
              
            elif self.norm == 'L2':
                delta_init = torch.zeros_like(x)
                s = h // 5
                sp_init = (h - s * 5) // 2
                vh = sp_init + 0
                for _ in range(h // s):
                    vw = sp_init + 0
                    for _ in range(w // s):
                        delta_init[:, :, vh:vh + s, vw:vw + s] += self.eta(
                            s).view(1, 1, s, s) * self.random_choice(
                            [x.shape[0], c, 1, 1])
                        vw += s
                    vh += s

                x_best = torch.clamp(x + self.normalize(delta_init
                    ) * self.eps, 0., 1.)
                margin_min, loss_min = self.margin_and_loss(x_best, y)
                n_queries = torch.ones(x.shape[0]).to(self.device)
                s_init = int(math.sqrt(self.p_init * n_features / c))

                for i_iter in range(self.n_queries):
                    idx_to_fool = (margin_min > 0.0).nonzero().squeeze()

                    x_curr = self.check_shape(x[idx_to_fool])
                    x_best_curr = self.check_shape(x_best[idx_to_fool])
                    y_curr = y[idx_to_fool]
                    if len(y_curr.shape) == 0:
                        y_curr = y_curr.unsqueeze(0)
                    margin_min_curr = margin_min[idx_to_fool]
                    loss_min_curr = loss_min[idx_to_fool]

                    delta_curr = x_best_curr - x_curr
                    p = self.p_selection(i_iter)
                    s = max(int(round(math.sqrt(p * n_features / c))), 3)
                    if s % 2 == 0:
                        s += 1

                    vh = self.random_int(0, h - s)
                    vw = self.random_int(0, w - s)
                    new_deltas_mask = torch.zeros_like(x_curr)
                    new_deltas_mask[:, :, vh:vh + s, vw:vw + s] = 1.0
                    norms_window_1 = (delta_curr[:, :, vh:vh + s, vw:vw + s
                        ] ** 2).sum(dim=(-2, -1), keepdim=True).sqrt()

                    vh2 = self.random_int(0, h - s)
                    vw2 = self.random_int(0, w - s)
                    new_deltas_mask_2 = torch.zeros_like(x_curr)
                    new_deltas_mask_2[:, :, vh2:vh2 + s, vw2:vw2 + s] = 1.

                    norms_image = self.lp_norm(x_best_curr - x_curr)
                    mask_image = torch.max(new_deltas_mask, new_deltas_mask_2)
                    norms_windows = self.lp_norm(delta_curr * mask_image)

                    new_deltas = torch.ones([x_curr.shape[0], c, s, s]
                        ).to(self.device)
                    new_deltas *= (self.eta(s).view(1, 1, s, s) *
                        self.random_choice([x_curr.shape[0], c, 1, 1]))
                    old_deltas = delta_curr[:, :, vh:vh + s, vw:vw + s] / (
                        1e-12 + norms_window_1)
                    new_deltas += old_deltas
                    new_deltas = new_deltas / (1e-12 + (new_deltas ** 2).sum(
                        dim=(-2, -1), keepdim=True).sqrt()) * (torch.max(
                        (self.eps * torch.ones_like(new_deltas)) ** 2 -
                        norms_image ** 2, torch.zeros_like(new_deltas)) /
                        c + norms_windows ** 2).sqrt()
                    delta_curr[:, :, vh2:vh2 + s, vw2:vw2 + s] = 0.
                    delta_curr[:, :, vh:vh + s, vw:vw + s] = new_deltas + 0

                    x_new = torch.clamp(x_curr + self.normalize(delta_curr
                        ) * self.eps, 0. ,1.)
                    x_new = self.check_shape(x_new)
                    norms_image = self.lp_norm(x_new - x_curr)

                    margin, loss = self.margin_and_loss(x_new, y_curr)

                    # update loss if new loss is better
                    idx_improved = (loss < loss_min_curr).float()

                    loss_min[idx_to_fool] = idx_improved * loss + (
                        1. - idx_improved) * loss_min_curr

                    # update margin and x_best if new loss is better
                    # or misclassification
                    idx_miscl = (margin <= 0.).float()
                    idx_improved = torch.max(idx_improved, idx_miscl)

                    margin_min[idx_to_fool] = idx_improved * margin + (
                        1. - idx_improved) * margin_min_curr
                    idx_improved = idx_improved.reshape([-1,
                        *[1]*len(x.shape[:-1])])
                    x_best[idx_to_fool] = idx_improved * x_new + (
                        1. - idx_improved) * x_best_curr
                    n_queries[idx_to_fool] += 1.

                    ind_succ = (margin_min <= 0.).nonzero().squeeze()
                    if self.verbose and ind_succ.numel() != 0:
                        print('{}'.format(i_iter + 1),
                            '- success rate={}/{} ({:.2%})'.format(
                            ind_succ.numel(), n_ex_total, float(
                            ind_succ.numel()) / n_ex_total),
                            '- avg # queries={:.1f}'.format(
                            n_queries[ind_succ].mean().item()),
                            '- med # queries={:.1f}'.format(
                            n_queries[ind_succ].median().item()),
                            '- loss={:.3f}'.format(loss_min.mean()))

                    assert (x_new != x_new).sum() == 0
                    assert (x_best != x_best).sum() == 0
                    
                    if ind_succ.numel() == n_ex_total:
                        break

        return n_queries, x_best

    def perturb(self, x, y=None):
        """
        :param x:           clean images
        :param y:           untargeted attack -> clean labels,
                            if None we use the predicted labels
                            targeted attack -> target labels, if None random classes,
                            different from the predicted ones, are sampled
        """

        self.init_hyperparam(x)

        adv = x.clone()
        if y is None:
            if not self.targeted:
                with torch.no_grad():
                    output = self.model(x)
                    y_pred = output.max(1)[1]
                    y = y_pred.detach().clone().long().to(self.device)
            else:
                with torch.no_grad():
                    output = self.model(x)
                    n_classes = output.shape[-1]
                    y_pred = output.max(1)[1]
                    y = self.random_target_classes(y_pred, n_classes)
        else:
            y = y.detach().clone().long().to(self.device)

        if not self.targeted:
            acc = self.model(x).max(1)[1] == y
        else:
            acc = self.model(x).max(1)[1] != y

        startt = time.time()

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)

        for counter in range(self.n_restarts):
            ind_to_fool = acc.nonzero().squeeze()
            if len(ind_to_fool.shape) == 0:
                ind_to_fool = ind_to_fool.unsqueeze(0)
            if ind_to_fool.numel() != 0:
                x_to_fool = x[ind_to_fool].clone()
                y_to_fool = y[ind_to_fool].clone()

                _, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)

                output_curr = self.model(adv_curr)
                if not self.targeted:
                    acc_curr = output_curr.max(1)[1] == y_to_fool
                else:
                    acc_curr = output_curr.max(1)[1] != y_to_fool
                ind_curr = (acc_curr == 0).nonzero().squeeze()

                acc[ind_to_fool[ind_curr]] = 0
                adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                if self.verbose:
                    print('restart {} - robust accuracy: {:.2%}'.format(
                        counter, acc.float().mean()),
                        '- cum. time: {:.1f} s'.format(
                        time.time() - startt))

        return adv