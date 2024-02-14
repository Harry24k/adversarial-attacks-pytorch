import torch
import torch.nn as nn

from ..attack import Attack


class EADL1(Attack):
    r"""
    EAD attack in the paper 'EAD: Elastic-Net Attacks to Deep Neural Networks'
    [https://arxiv.org/abs/1709.04114]

    Distance Measure : L1

    Arguments:
        model (nn.Module): model to attack.
        init_c(float): the initial constant c to pick as a first guess. (Default: 1)
        kappa (float): how strong the adversarial example should be (also written as 'confidence'). (Default: 0)
        beta (float): hyperparameter trading off L2 minimization for L1 minimization. (Default: 0.001)
        steps (int): number of iterations to perform gradient descent. (Default: 10)
        lr (float): larger values converge faster to less accurate results. (Default: 0.01)
        binary_search_steps (int): number of times to adjust the constant with binary search. (Default: 9)
        abort_early (bool): if we stop improving, abort gradient descent early. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.EADL1(model, init_c=1, kappa=0, steps=10, lr=0.01)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        model,
        init_c=1,
        kappa=0,
        beta=0.001,
        steps=10,
        lr=0.01,
        binary_search_steps=9,
        abort_early=True,
    ):
        super().__init__("EADL1", model)
        self.init_c = init_c
        self.kappa = kappa
        self.beta = beta
        self.steps = steps
        self.lr = lr
        self.binary_search_steps = binary_search_steps
        self.abort_early = abort_early
        # The last iteration (if we run many steps) repeat the search once.
        self.repeat = binary_search_steps >= 10
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            labels = self.get_target_label(images, labels)

        outputs = self.get_logits(images)

        batch_size = images.shape[0]
        lower_bound = torch.zeros((batch_size, )).to(self.device)
        const = torch.ones(batch_size).to(self.device) * self.init_c
        upper_bound = torch.ones(batch_size).to(self.device) * 1e10

        o_best_adv_images = images.clone()

        o_best_score = torch.full(
            (batch_size, ), -1, dtype=torch.long).to(self.device)
        o_best_L1 = torch.full((batch_size, ), 1e10).to(self.device)
        # Initialization: x^{(0)} = y^{(0)} = x_0 in paper Algorithm 1 part
        x_k = images.clone().detach()
        y_k = nn.Parameter(images)

        # Start binary search
        for outer_step in range(self.binary_search_steps):
            self.global_step = 0
            best_score = torch.full(
                (batch_size, ), -1, dtype=torch.long).to(self.device)
            best_L1 = torch.full((batch_size, ), 1e10).to(self.device)

            if self.repeat and outer_step == (self.binary_search_steps - 1):
                const = upper_bound

            lr = self.lr
            prev_cost = 1e6
            for step in range(self.steps):
                # reset gradient
                if y_k.grad is not None:
                    y_k.grad.detach_()
                    y_k.grad.zero_()

                # Loss over images_parameters with only L2 same as CW
                # We don't update L1 loss with SGD because we use ISTA
                outputs = self.get_logits(y_k)
                L2_loss = self.L2_loss(y_k, images)

                cost = self.EAD_loss(outputs, labels, None, L2_loss, const)
                cost.backward()

                # Gradient step
                self.global_step += 1
                with torch.no_grad():
                    y_k -= y_k.grad * lr

                # Ploynomial decay of learning rate
                lr = self.lr * (1 - self.global_step / self.steps) ** 0.5
                x_k, y_k = self.FISTA(images, x_k, y_k)
                # Loss ElasticNet or L1 over x_k
                with torch.no_grad():
                    outputs = self.get_logits(x_k)
                    L2_loss = self.L2_loss(x_k, images)
                    L1_loss = self.L1_loss(x_k, images)
                    cost = self.EAD_loss(
                        outputs, labels, L1_loss, L2_loss, const)

                    # L1 attack key step!
                    current_Lx = L1_loss

                    # Update adversarial images
                    pre = torch.argmax(outputs.detach(), 1)
                    condition_1 = self.compare(pre, labels)
                    condition_2 = (current_Lx < best_L1)
                    # Filter out images that get either correct predictions or non-decreasing loss,
                    # i.e., only images that are both misclassified and loss-decreasing are left
                    mask_1_2 = torch.logical_and(condition_1, condition_2)
                    best_L1[mask_1_2] = current_Lx[mask_1_2]
                    best_score[mask_1_2] = pre[mask_1_2]

                    condition_3 = (current_Lx < o_best_L1)
                    o_mask = torch.logical_and(condition_1, condition_3)
                    o_best_L1[o_mask] = current_Lx[o_mask]
                    o_best_score[o_mask] = pre[o_mask]

                    o_best_adv_images[o_mask] = x_k[o_mask]

                    # print('loss: {}, prevloss: {}'.format(loss, prevloss))
                    if self.abort_early and step % (self.steps // 10) == 0:
                        if cost > prev_cost * 0.9999:
                            break
                        prev_cost = cost

            # Adjust the constant as needed
            outputs = self.get_logits(x_k)
            pre = torch.argmax(outputs, 1)

            condition_1 = self.compare(pre, labels)
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

        return o_best_adv_images

    def compare(self, predition, labels):
        if self.targeted:
            # We want to let pre == target_labels in a targeted attack
            ret = (predition == labels)
        else:
            # If the attack is not targeted we simply make these two values unequal
            ret = (predition != labels)
        return ret

    def L1_loss(self, x1, x2):
        Flatten = nn.Flatten()
        L1_loss = torch.abs(Flatten(x1) - Flatten(x2)).sum(dim=1)
        return L1_loss

    def L2_loss(self, x1, x2):
        MSELoss = nn.MSELoss(reduction="none")
        Flatten = nn.Flatten()
        L2_loss = MSELoss(Flatten(x1), Flatten(x2)).sum(dim=1)
        return L2_loss

    def EAD_loss(self, outputs, labels, L1_loss, L2_loss, const):
        one_hot_labels = torch.eye(outputs.shape[1]).to(self.device)[labels]

        # get the target class's logit
        real = torch.sum(one_hot_labels * outputs, dim=1)
        # find the max logit other than the target classs
        other = torch.max((1 - one_hot_labels) * outputs - (one_hot_labels * 1e12), dim=1)[0]  # nopep8

        if self.targeted:
            F_loss = torch.clamp((other - real), min=-self.kappa)
        else:
            F_loss = torch.clamp((real - other), min=-self.kappa)

        if isinstance(L1_loss, type(None)):
            loss = torch.sum(const * F_loss) + torch.sum(L2_loss)
        else:
            loss = torch.sum(const * F_loss) + torch.sum(L2_loss) + torch.sum(self.beta * L1_loss)  # nopep8

        return loss

    def FISTA(self, images, x_k, y_k):

        zt = self.global_step / (self.global_step + 3)

        upper = torch.clamp(y_k - self.beta, max=1)
        lower = torch.clamp(y_k + self.beta, min=0)

        diff = y_k - images
        c1 = diff > self.beta
        c2 = torch.abs(diff) <= self.beta
        c3 = diff < -self.beta

        new_x_k = x_k.clone().detach().to(self.device)
        new_x_k = (c1.float() * upper) + (c2.float() * images) + (c3.float() * lower)  # nopep8
        y_k.data = new_x_k + (zt * (new_x_k - x_k))
        return new_x_k, y_k
