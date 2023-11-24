import torch
import torch.nn as nn
import torch.optim as optim

from ..attack import Attack


class CWL2(Attack):
    r"""
    CW in the paper 'Towards Evaluating the Robustness of Neural Networks'
    [https://arxiv.org/abs/1608.04644]

    Distance Measure : L2

    Arguments:
        model (nn.Module): model to attack.
        c (float): c in the paper. parameter for box-constraint. (Default: 1)    
            :math:`minimize \Vert\frac{1}{2}(tanh(w)+1)-x\Vert^2_2+c\cdot f(\frac{1}{2}(tanh(w)+1))`
        kappa (float): kappa (also written as 'confidence') in the paper. (Default: 0)
            :math:`f(x')=max(max\{Z(x')_i:i\neq t\} -Z(x')_t, - \kappa)`
        steps (int): number of steps (also written as 'max_iterations'). (Default: 50)
        lr (float): learning rate of the Adam optimizer. (Default: 0.01)
        abort_early: if true, allows early aborts if gradient descent gets stuck. (Default: True)

    .. warning:: With default c, you can't easily get adversarial images. Set higher c like 1.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.CWL2(model, c=1, kappa=0, steps=50, lr=0.01, abort_early=True)
        >>> adv_images = attack(images, labels)

    .. note:: The binary search version of the CW algorithm has been implemented as CWBS.

    """

    def __init__(self, model, c=1, kappa=0, steps=50, lr=0.01, abort_early=True):
        super().__init__("CWL2", model)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.abort_early = abort_early
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)
        else:
            target_labels = labels

        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        batch_size = len(images)
        best_Lx = torch.full((batch_size, ), 1e10).to(self.device)
        prev_cost = 1e10

        MSELoss = nn.MSELoss(reduction="none")
        Flatten = nn.Flatten()

        optimizer = optim.Adam([w], lr=self.lr)

        for step in range(self.steps):
            # Get adversarial images
            adv_images = self.tanh_space(w)

            # Calculate loss
            current_Lx = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)

            Lx_loss = current_Lx.sum()

            outputs = self.get_logits(adv_images)
            if self.targeted:
                f_loss = self.f(outputs, target_labels)
            else:
                f_loss = self.f(outputs, labels)

            cost = Lx_loss + torch.sum(self.c * f_loss)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update adversarial images
            pre = torch.argmax(outputs.detach(), 1)
            condition_1 = self.compare(pre, labels, target_labels)
            condition_2 = (current_Lx < best_Lx)

            # Filter out images that get either correct predictions or non-decreasing loss,
            # i.e., only images that are both misclassified and loss-decreasing are left
            mask = torch.logical_and(condition_1, condition_2)
            best_Lx[mask] = current_Lx[mask]
            best_adv_images[mask] = adv_images[mask]

            # Early stop when loss does not converge
            if self.abort_early and step % (self.steps // 10) == 0:
                if cost > prev_cost * 0.9999:
                    break
                else:
                    prev_cost = cost

        # print(best_Lx)
        return best_adv_images

    def compare(self, predition, labels, target_labels):
        if self.targeted:
            # We want to let pre == target_labels in a targeted attack
            ret = (predition == target_labels)
        else:
            # If the attack is not targeted we simply make these two values unequal
            ret = (predition != labels)
        return ret

    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        # atanh is defined in the range -1 to 1
        return self.atanh(torch.clamp(x * 2 - 1, min=-1, max=1))

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(outputs.shape[1]).to(self.device)[labels]

        # get the target class's logit
        real = torch.sum(one_hot_labels * outputs, dim=1)
        # find the max logit other than the target class
        other = torch.max((1 - one_hot_labels) * outputs - one_hot_labels * 1e12, dim=1)[0]  # nopep8

        if self.targeted:
            return torch.clamp((other - real), min=-self.kappa)
        else:
            return torch.clamp((real - other), min=-self.kappa)
