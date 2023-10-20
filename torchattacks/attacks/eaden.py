import torch
import torch.nn as nn

from ..attack import Attack


class EADEN(Attack):
    r"""
    EAD attack in the paper 'EAD: Elastic-Net Attacks to Deep Neural Networks'
    [https://arxiv.org/abs/1709.04114]

    Distance Measure : L1 and L2

    Arguments:
        model (nn.Module): model to attack.
        kappa (float): how strong the adversarial example should be (also written as 'confidence'). (Default: 0)
        lr (float): larger values converge faster to less accurate results. (Default: 0.01)
        binary_search_steps (int): number of times to adjust the constant with binary search. (Default: 9)
        max_iterations (int): number of iterations to perform gradient descent. (Default: 100)
        abort_early (bool): if we stop improving, abort gradient descent early. (Default: True)
        initial_const (float): the initial constant c to pick as a first guess. (Default: 0.001)
        beta (float): hyperparameter trading off L2 minimization for L1 minimization. (Default: 0.001)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.EADEN(model, kappa=0, lr=0.01, max_iterations=100)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        model,
        kappa=0,
        lr=0.01,
        binary_search_steps=9,
        max_iterations=100,
        abort_early=True,
        initial_const=0.001,
        beta=0.001,
    ):
        super().__init__("EADEN", model)
        self.kappa = kappa
        self.lr = lr
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.beta = beta
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
        lower_bound = torch.zeros(batch_size, device=self.device)
        const = torch.ones(batch_size, device=self.device) * self.initial_const
        upper_bound = torch.ones(batch_size, device=self.device) * 1e10

        final_adv_images = images.clone()
        y_one_hot = torch.eye(outputs.shape[1]).to(self.device)[labels]

        o_bestl1 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestl1 = torch.Tensor(o_bestl1).to(self.device)
        o_bestscore = torch.Tensor(o_bestscore).to(self.device)

        # Initialization: x^{(0)} = y^{(0)} = x_0 in paper Algorithm 1 part
        x_k = images.clone().detach()
        y_k = nn.Parameter(images)

        # Start binary search
        for outer_step in range(self.binary_search_steps):

            self.global_step = 0

            bestl1 = [1e10] * batch_size
            bestscore = [-1] * batch_size

            bestl1 = torch.Tensor(bestl1).to(self.device)
            bestscore = torch.Tensor(bestscore).to(self.device)
            prevloss = 1e6

            if self.repeat and outer_step == (self.binary_search_steps - 1):
                const = upper_bound

            lr = self.lr
            for iteration in range(self.max_iterations):
                # reset gradient
                if y_k.grad is not None:
                    y_k.grad.detach_()
                    y_k.grad.zero_()

                # Loss over images_parameters with only L2 same as CW
                # we don't update L1 loss with SGD because we use ISTA
                output = self.get_logits(y_k)
                L2_loss = self.L2_loss(y_k, images)

                cost = self.EAD_loss(output, y_one_hot, None, L2_loss, const)
                # cost.backward(retain_graph=True)
                cost.backward()

                # Gradient step
                # y_k.data.add_(-lr, y_k.grad.data)
                self.global_step += 1
                with torch.no_grad():
                    y_k -= y_k.grad * lr

                # Ploynomial decay of learning rate
                lr = (
                    self.lr * (1 - self.global_step / self.max_iterations) ** 0.5
                )  # nopep8
                x_k, y_k = self.FISTA(images, x_k, y_k)
                # Loss ElasticNet or L1 over x_k
                with torch.no_grad():
                    output = self.get_logits(x_k)
                    L2_loss = self.L2_loss(x_k, images)
                    L1_loss = self.L1_loss(x_k, images)
                    loss = self.EAD_loss(
                        output, y_one_hot, L1_loss, L2_loss, const
                    )  # nopep8

                    # print('loss: {}, prevloss: {}'.format(loss, prevloss))
                    if (
                        self.abort_early
                        and iteration % (self.max_iterations // 10) == 0
                    ):
                        if loss > prevloss * 0.999999:
                            break
                        prevloss = loss

                    # EN attack key step!
                    cost = L2_loss + (L1_loss * self.beta)
                    self.adjust_best_result(
                        x_k,
                        labels,
                        output,
                        cost,
                        bestl1,
                        bestscore,
                        o_bestl1,
                        o_bestscore,
                        final_adv_images,
                    )

            self.adjust_constant(labels, bestscore, const, upper_bound, lower_bound)

        return final_adv_images

    def L1_loss(self, x1, x2):
        Flatten = nn.Flatten()
        L1_loss = torch.abs(Flatten(x1) - Flatten(x2)).sum(dim=1)
        # L1_loss = L1.sum()
        return L1_loss

    def L2_loss(self, x1, x2):
        MSELoss = nn.MSELoss(reduction="none")
        Flatten = nn.Flatten()
        L2_loss = MSELoss(Flatten(x1), Flatten(x2)).sum(dim=1)
        # L2_loss = L2.sum()
        return L2_loss

    def EAD_loss(self, output, one_hot_labels, L1_loss, L2_loss, const):

        # Not same as CW's f function
        other = torch.max(
            (1 - one_hot_labels) * output - (one_hot_labels * 1e4), dim=1
        )[0]
        real = torch.max(one_hot_labels * output, dim=1)[0]

        if self.targeted:
            F_loss = torch.clamp((other - real), min=-self.kappa)
        else:
            F_loss = torch.clamp((real - other), min=-self.kappa)

        if isinstance(L1_loss, type(None)):
            loss = torch.sum(const * F_loss) + torch.sum(L2_loss)
        else:
            loss = (
                torch.sum(const * F_loss)
                + torch.sum(L2_loss)
                + torch.sum(self.beta * L1_loss)
            )

        return loss

    def FISTA(self, images, x_k, y_k):

        zt = self.global_step / (self.global_step + 3)

        upper = torch.clamp(y_k - self.beta, max=1)
        lower = torch.clamp(y_k + self.beta, min=0)

        diff = y_k - images
        cond1 = (diff > self.beta).float()
        cond2 = (torch.abs(diff) <= self.beta).float()
        cond3 = (diff < -self.beta).float()

        new_x_k = (cond1 * upper) + (cond2 * images) + (cond3 * lower)
        y_k.data = new_x_k + (zt * (new_x_k - x_k))
        return new_x_k, y_k

    def compare(self, output, labels):
        if len(output.shape) >= 2:
            # output is tensor
            output = output.clone().detach()
            if self.targeted:
                output[:, labels] -= self.kappa
            else:
                output[:, labels] += self.kappa
            output = torch.argmax(output, 1)
        else:
            # output is int or float
            pass

        if self.targeted:
            return output == labels
        else:
            return output != labels

    def adjust_best_result(
        self,
        adv_img,
        labels,
        output,
        cost,
        bestl1,
        bestscore,
        o_bestl1,
        o_bestscore,
        final_adv_images,
    ):
        output_label = torch.argmax(output, 1).float()
        mask = (cost < bestl1) & self.compare(output, labels)
        bestl1[mask] = cost[mask]
        bestscore[mask] = output_label[mask]

        mask = (cost < o_bestl1) & self.compare(output, labels)
        o_bestl1[mask] = cost[mask]
        o_bestscore[mask] = output_label[mask]
        final_adv_images[mask] = adv_img[mask]

    def adjust_constant(self, labels, bestscore, const, upper_bound, lower_bound):
        mask = (self.compare(bestscore, labels)) & (bestscore != -1)
        upper_bound[mask] = torch.min(upper_bound[mask], const[mask])
        lower_bound[~mask] = torch.max(lower_bound[~mask], const[~mask])  # nopep8

        mask = upper_bound < 1e9
        const[mask] = (lower_bound[mask] + upper_bound[mask]) / 2
        const[~mask] = const[~mask] * 10
