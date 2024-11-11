import torch
import torch.distributions.uniform as uniform

from ..attack import Attack


class FAB(Attack):
    r"""
    Fast Adaptive Boundary Attack (FAB) in the paper 'Minimally distorted Adversarial Examples with a Fast Adaptive Boundary Attack'
    [https://arxiv.org/abs/1907.02044]
    [https://github.com/fra31/fab-attack]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        n_restarts (int): number of random restarts. (Default: 1)
        n_iter (int): number of steps. (Default: 10)
        alpha_max (float): alpha_max. (Default: 0.1)
        eta (float): overshooting. (Default: 1.05)
        beta (float): backward step. (Default: 0.9)
        las (bool): final search. (Default: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FAB(model, eps=8/255, n_restarts=1, n_iter=10, alpha_max=0.1, eta=1.05, beta=0.9, las=False)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8/255, n_restarts=1, n_iter=10, alpha_max=0.1, eta=1.05, beta=0.9, las=False):
        super().__init__("FAB", model)
        self.eps = eps
        self.n_restarts = n_restarts
        self.n_iter = n_iter
        self.alpha_max = alpha_max
        self.eta = eta
        self.beta = beta
        self.las = las
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self.perturb(images, labels)

        return adv_images

    def perturb(self, images, labels):
        logits = self.get_logits(images)
        pred = torch.argmax(logits, 1) == labels
        pred1 = pred.clone()
        im2 = images[pred].clone()
        la2 = labels[pred].clone()
        bs = torch.sum(pred)
        u1 = torch.arange(bs)

        adv = im2.clone()
        adv_c = images.clone()
        res2 = torch.full((bs, ), 1e10, device=self.device)
        x1 = torch.clone(im2)
        x0 = torch.clone(im2).reshape(bs, -1)
        eps = torch.full(res2.shape, self.eps, device=self.device)

        if self.targeted:
            # The code provided in the original paper does not implement the target attack code,
            # and the code here is implemented based on the relevant code in the original author's subsequent autoattack work.
            # https://github.com/fra31/auto-attack
            target_labels = self.get_target_label(images, labels)
            la_target2 = target_labels[pred].detach().clone()
        else:
            la_target2 = None

        for counter_restarts in range(self.n_restarts):
            if counter_restarts > 0:
                t = uniform.Uniform(-1, 1).sample(x1.shape).to(self.device)
                a = torch.min(res2, eps).reshape((-1, 1, 1, 1)) * t
                b = torch.abs(t.view(t.shape[0], -1)).max(dim=1, keepdim=True)[0].view((-1, 1, 1, 1))  # nopep8
                x1 = im2 + a / b * 0.5
                x1 = torch.clamp(x1, min=0.0, max=1.0)

            for _ in range(self.n_iter):
                # print(i)
                df, dg = self.get_diff_logits_grads_batch(x1, la2, la_target2)
                dist1 = torch.abs(df) / (1e-8 + torch.sum(torch.abs(dg).view(dg.shape[0], dg.shape[1], -1), -1))  # nopep8
                ind = torch.argmin(dist1, 1)
                b = - df[u1, ind] + torch.sum(torch.reshape(dg[u1, ind] * x1, (bs, -1)), 1).to(self.device)  # nopep8
                w = torch.reshape(dg[u1, ind], [bs, -1]).to(self.device)
                x2 = torch.reshape(x1, (bs, -1))
                d3 = self.projection_linf(torch.cat((x2, x0), 0), torch.cat((w, w), 0), torch.cat((b, b), 0))  # nopep8
                d1 = torch.reshape(d3[:bs], x1.shape)
                d2 = torch.reshape(d3[-bs:], x1.shape)
                a0 = torch.abs(d3).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)  # nopep8
                a0 = torch.max(a0, 1e-8 * torch.ones(a0.shape, device=self.device))  # nopep8
                a1 = a0[:bs]
                a2 = a0[-bs:]
                temp_var_1 = torch.max(a1 / (a1 + a2), torch.zeros(a1.shape, device=self.device))  # nopep8
                temp_var_2 = self.alpha_max * torch.ones(a1.shape, device=self.device)  # nopep8
                alpha = torch.min(temp_var_1, temp_var_2)
                x1 = torch.clamp((x1 + self.eta * d1) * (1 - alpha) + (im2 + d2 * self.eta) * alpha, min=0.0, max=1.0)  # nopep8
                is_adv = torch.argmax(self.get_logits(x1), 1) != la2
                if torch.sum(is_adv) > 0:
                    temp_var = torch.reshape(x1[is_adv] - im2[is_adv], (torch.sum(is_adv), -1))  # nopep8
                    t = torch.max(torch.abs(temp_var), 1)[0]
                    temp_var_3 = x1[is_adv] * (t < res2[is_adv]).float().reshape([-1, 1, 1, 1])  # nopep8
                    temp_var_4 = adv[is_adv] * (t >= res2[is_adv]).float().reshape([-1, 1, 1, 1])  # nopep8
                    adv[is_adv] = temp_var_3 + temp_var_4
                    res2[is_adv] = t * (t < res2[is_adv]).float() + res2[is_adv] * (t >= res2[is_adv]).float()  # nopep8
                    x1[is_adv] = im2[is_adv] + (x1[is_adv] - im2[is_adv]) * self.beta  # nopep8

        if self.las:
            adv = self.linear_approximation_search(im2, la2, adv, 3)

        adv_c[pred1] = adv
        return adv_c

    def get_diff_logits_grads_batch(self, images, labels, target_labels=None):
        images = images.clone().detach().requires_grad_()  # make sure its was leaf node
        # print(images.is_leaf)

        if not self.targeted:
            logits = self.get_logits(images)
            g2 = self.compute_jacobian(images, logits)
            y2 = logits
            df = y2 - torch.unsqueeze(y2[torch.arange(images.shape[0]), labels], 1)  # nopep8
            dg = g2 - torch.unsqueeze(g2[torch.arange(images.shape[0]), labels], 1)  # nopep8
            df[torch.arange(images.shape[0]), labels] = 1e10
        else:
            u = torch.arange(images.shape[0])
            logits = self.get_logits(images)
            diff_logits = -(logits[u, labels] - logits[u, target_labels])
            sum_diff = torch.sum(diff_logits)

            # jacobian
            self.zero_gradients(images)
            sum_diff.backward()
            grad_diff = images.grad.data
            df = torch.unsqueeze(diff_logits.detach(), 1)
            dg = torch.unsqueeze(grad_diff, 1)

        return df, dg

    def compute_jacobian(self, images, logits):
        num_classes = logits.shape[1]
        jacobian = torch.zeros(num_classes, *images.size()).to(self.device)
        grad_output = torch.zeros_like(logits).to(self.device)

        for i in range(num_classes):
            self.zero_gradients(images)
            grad_output.zero_()
            grad_output[:, i] = 1
            logits.backward(grad_output, retain_graph=True)
            jacobian[i] = images.grad.data

        return torch.transpose(jacobian, dim0=0, dim1=1)

    def zero_gradients(self, x):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()

    def projection_linf(self, t2, w2, b2):
        t = t2.clone().float()
        w = w2.clone().float()
        b = b2.clone().float()

        ind2 = torch.nonzero(torch.sum(w * t, 1) - b < 0)
        w[ind2] *= -1
        b[ind2] *= -1

        c5 = (w < 0).type(torch.FloatTensor).to(self.device)
        a = torch.ones(t.shape).to(self.device)
        d = (a * c5 - t) * (w != 0).type(torch.FloatTensor).to(self.device)
        a -= a * (1 - c5)

        p = torch.ones(t.shape, device=self.device) * c5 - t * (2 * c5 - 1)
        indp = torch.argsort(p, dim=1)

        b = b - torch.sum(w * t, 1)
        b0 = torch.sum(w * d, 1)
        b1 = b0.clone()

        indp2 = indp.unsqueeze(-1).flip(dims=(1, 2)).squeeze()
        u = torch.arange(0, w.shape[0])
        ws = w[u.unsqueeze(1), indp2]
        bs2 = -ws * d[u.unsqueeze(1), indp2]

        s = torch.cumsum(ws.abs(), dim=1)
        sb = torch.cumsum(bs2, dim=1) + b0.unsqueeze(1)

        b2 = sb[u, -1] - s[u, -1] * p[u, indp[u, 0]]
        c_l = torch.nonzero(b - b2 > 0).squeeze()
        c2 = torch.nonzero((b - b1 > 0) * (b - b2 <= 0)).squeeze()

        lb = torch.zeros(c2.shape[0], device=self.device)
        ub = torch.ones(c2.shape[0], device=self.device) * (w.shape[1] - 1)
        nitermax = torch.ceil(torch.log2(torch.tensor(w.shape[1]).float()))

        for _ in range(int(nitermax.item())):
            counter4 = torch.floor((lb + ub) / 2)
            counter2 = counter4.type(torch.LongTensor)
            indcurr = indp[c2, -counter2 - 1]
            b2 = sb[c2, counter2] - s[c2, counter2] * p[c2, indcurr]
            ind3 = b[c2] - b2 > 0
            ind32 = ~ind3
            lb[ind3] = counter4[ind3]
            ub[ind32] = counter4[ind32]

        lb = lb.cpu().numpy().astype(int)

        if c_l.nelement != 0:
            m = torch.max((b[c_l] - sb[c_l, -1]) / (-s[c_l, -1]), torch.zeros(sb[c_l, -1].shape, device=self.device))  # nopep8
            lmbd_opt = torch.unsqueeze(m, -1)
            d[c_l] = (2 * a[c_l] - 1) * lmbd_opt

        m = torch.max((b[c2] - sb[c2, lb])/(-s[c2, lb]), torch.zeros(sb[c2, lb].shape, device=self.device))  # nopep8
        lmbd_opt = torch.unsqueeze(m, -1)
        d[c2] = torch.min(lmbd_opt, d[c2]) * c5[c2] + torch.max(-lmbd_opt, d[c2]) * (1 - c5[c2])  # nopep8

        return d * (w != 0).type(torch.FloatTensor).to(self.device)

    def linear_approximation_search(self, clean_images, clean_labels, adv_images, niter):
        a1 = clean_images.clone()
        a2 = adv_images.clone()
        u = torch.arange(clean_images.shape[0])
        y1 = self.get_logits(a1)
        y2 = self.get_logits(a2)
        la2 = torch.argmax(y2, 1)

        for _ in range(niter):
            t1 = (y1[u, clean_labels] - y1[u, la2]).reshape([-1, 1, 1, 1])
            t2 = (-(y2[u, clean_labels] - y2[u, la2])).reshape([-1, 1, 1, 1])

            t3 = t1 / (t1 + t2 + 1e-10)
            c3 = torch.logical_and(0.0 <= t3, t3 <= 1.0)
            t3[~c3] = 1.0

            a3 = a1 * (1.0 - t3) + a2 * t3
            y3 = self.get_logits(a3)
            la3 = torch.argmax(y3, 1)
            pred = la3 == clean_labels

            y1[pred] = y3[pred] + 0
            a1[pred] = a3[pred] + 0
            y2[~pred] = y3[~pred] + 0
            la2[~pred] = la3[~pred] + 0
            a2[~pred] = a3[~pred] + 0

        return a2
