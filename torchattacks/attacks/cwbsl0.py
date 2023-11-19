import torch
import torch.nn as nn
import torch.optim as optim

from ..attack import Attack


class CWBSL0(Attack):
    r"""
    CW (binary search version) in the paper 'Towards Evaluating the Robustness of Neural Networks'
    [https://arxiv.org/abs/1608.04644]

    Distance Measure : L0

    Arguments:
        model (nn.Module): model to attack.
        init_c (float): init_c (or c) in the paper. parameter for box-constraint. (Default: 1)    
            :math:`minimize \Vert\frac{1}{2}(tanh(w)+1)-x\Vert^2_2+c\cdot f(\frac{1}{2}(tanh(w)+1))`
        kappa (float): kappa (also written as 'confidence') in the paper. (Default: 0)
            :math:`f(x')=max(max\{Z(x')_i:i\neq t\} -Z(x')_t, - \kappa)`
        steps (int): number of steps (also written as 'max_iterations'). (Default: 50)
        lr (float): learning rate of the Adam optimizer. (Default: 0.01)
        binary_search_steps (int): The number of times we perform binary search to find the optimal tradeoff-constant between distance and confidence. (Default: 9)
        abort_early: if true, allows early aborts if gradient descent gets stuck. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.CWBSL0(model, init_c=1, kappa=0, steps=50, lr=0.01, binary_search_steps=9, abort_early=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, init_c=1, kappa=0, steps=50, lr=0.01, binary_search_steps=9, abort_early=True):
        super().__init__("CWBSL0", model)
        self.init_c = init_c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.binary_search_steps = binary_search_steps
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

        o_best_adv_images = images.clone().detach()

        optimizer = optim.Adam([w], lr=self.lr)

        batch_size = len(images)
        lower_bound = torch.zeros((batch_size, )).to(self.device)
        const = torch.full((batch_size, ), self.init_c,
                           dtype=torch.float).to(self.device)
        upper_bound = torch.full((batch_size, ), 1e10).to(self.device)

        o_best_score = torch.full(
            (batch_size, ), -1, dtype=torch.long).to(self.device)
        o_best_Lx = torch.full((batch_size, ), 1e10).to(self.device)

        threshold = 1e-6

        for _ in range(self.binary_search_steps):
            best_score = torch.full(
                (batch_size, ), -1, dtype=torch.long).to(self.device)
            best_Lx = torch.full((batch_size, ), 1e10).to(self.device)
            prev_cost = 1e10
            for step in range(self.steps):
                # Get adversarial images
                adv_images = self.tanh_space(w)

                # Calculate loss
                l0_norm = torch.abs(adv_images.reshape(-1) - images.reshape(-1))
                l0_condition = (l0_norm > threshold)
                # Number of non-zero values
                l0_value = (1.0 / l0_norm.shape[0]) * torch.sum(l0_condition)
                current_Lx = torch.full((batch_size, ), l0_value).to(self.device)

                Lx_loss = current_Lx.sum()

                outputs = self.get_logits(adv_images)
                if self.targeted:
                    # f_loss = self.f(outputs, target_labels).sum()
                    f_loss = self.f(outputs, target_labels)
                else:
                    # f_loss = self.f(outputs, labels).sum()
                    f_loss = self.f(outputs, labels)

                cost = Lx_loss + torch.sum(const * f_loss)

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                # Update adversarial images
                pre = torch.argmax(outputs.detach(), 1)
                condition_1 = self.compare(pre, labels, target_labels)
                condition_2 = (current_Lx < best_Lx)
                # Filter out images that get either correct predictions or non-decreasing loss,
                # i.e., only images that are both misclassified and loss-decreasing are left
                mask_1_2 = torch.logical_and(condition_1, condition_2)
                best_Lx[mask_1_2] = current_Lx[mask_1_2]
                best_score[mask_1_2] = pre[mask_1_2]

                condition_3 = (current_Lx < o_best_Lx)
                o_mask = torch.logical_and(condition_1, condition_3)
                o_best_Lx[o_mask] = current_Lx[o_mask]
                o_best_score[o_mask] = pre[o_mask]

                o_best_adv_images[o_mask] = adv_images[o_mask]

                # Check if we should abort search if we're getting nowhere.
                if self.abort_early and step % (self.steps // 10) == 0:
                    if cost > prev_cost * 0.9999:
                        break
                    else:
                        prev_cost = cost

            # Adjust the constant as needed
            outputs = self.get_logits(adv_images)
            pre = torch.argmax(outputs, 1)

            condition_1 = self.compare(pre, labels, target_labels)
            condition_2 = (best_score != -1)
            condition_3 = upper_bound < 1e9

            mask_1_2 = torch.logical_and(condition_1, condition_2)
            mask_1_2_3 = torch.logical_and(mask_1_2, condition_3)
            const_1 = (lower_bound + upper_bound) / 2.0

            upper_bound_min = torch.min(upper_bound, const)
            upper_bound[mask_1_2] = upper_bound_min[mask_1_2]
            const[mask_1_2_3] = const_1[mask_1_2_3]

            mask_n1_n2_3 = torch.logical_and(~mask_1_2, condition_3)
            upper_bound_max = torch.max(lower_bound, const)
            upper_bound[~mask_1_2] = upper_bound_max[~mask_1_2]
            const[mask_n1_n2_3] *= 10

        # print(o_best_Lx)
        return o_best_adv_images

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
        # find the max logit other than the target classs
        other = torch.max((1 - one_hot_labels) * outputs, dim=1)[0]

        if self.targeted:
            return torch.clamp((other - real), min=-self.kappa)
        else:
            return torch.clamp((real - other), min=-self.kappa)
