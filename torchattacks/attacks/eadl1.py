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
        kappa (float): (also written as 'confidence') how strong the adversarial example should be.
        lr (float): larger values converge faster to less accurate results.
        binary_search_steps (int): number of times to adjust the constant with binary search.
        max_iterations (int): number of iterations to perform gradient descent.
        abort_early (bool): if we stop improving, abort gradient descent early.
        initial_const (float): the initial constant c to pick as a first guess.
        beta (float): hyperparameter trading off L2 minimization for L1 minimization.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.EADL1(model, kappa=0, lr=0.01, max_iterations=100)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, kappa=0, lr=0.01, binary_search_steps=9, max_iterations=10000, abort_early=True, initial_const=0.001, beta=0.001):
        super().__init__("EADL1", model)
        self.kappa = kappa
        self.lr = lr
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.beta = beta
        # The last iteration (if we run many steps) repeat the search once.
        self.repeat = binary_search_steps >= 10
        self.supported_mode = ['default', 'targeted']

    def L1_loss(self, x1, x2):
        Flatten = nn.Flatten()
        L1_loss = torch.abs(Flatten(x1)-Flatten(x2)).sum(dim=1)
        # L1_loss = L1.sum()
        return L1_loss

    def L2_loss(self, x1, x2):
        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()
        L2_loss = MSELoss(Flatten(x1), Flatten(x2)).sum(dim=1)
        # L2_loss = L2.sum()
        return L2_loss

    def EAD_loss(self, output, one_hot, L1_loss, L2_loss, const):

        # Same as CW's f function
        other = torch.max((1-one_hot)*output-(one_hot*1e4), dim=1)[0]
        real = torch.max(one_hot*output, dim=1)[0]

        if self.targeted:
            F_loss = torch.clamp((other-real), min=-self.kappa)
        else:
            F_loss = torch.clamp((real-other), min=-self.kappa)

        if isinstance(L1_loss, type(None)):
            loss = torch.sum(const * F_loss) + torch.sum(L2_loss)
        else:
            loss = torch.sum(const * F_loss) + torch.sum(L2_loss) + torch.sum(self.beta * L1_loss)  # nopep8

        return loss

    def fast_ISTA(self, images, images_parameter, images_clone):

        zt = self.global_step / (self.global_step + 3)

        upper = torch.clamp(images_parameter - self.beta, max=1)
        lower = torch.clamp(images_parameter + self.beta, min=0)

        diff = images_parameter - images
        cond1 = (diff > self.beta).float()
        cond2 = (torch.abs(diff) <= self.beta).float()
        cond3 = (diff < -self.beta).float()

        new_images_clone = (cond1 * upper) + (cond2 * images) + (cond3 * lower)
        images_parameter.data = new_images_clone + (zt * (new_images_clone - images_clone))  # nopep8
        return images_parameter, new_images_clone

    def _is_successful(self, y1, y2):
        if self.targeted:
            return y1 == y2
        else:
            return y1 != y2

    def is_successful(self, output, label, is_logits):
        # Determine success, see if confidence-adjusted logits give the right label
        if is_logits:
            output = output.detach().clone()
            if self.targeted:
                output[torch.arange(len(label)).long(),
                       label] -= self.kappa
            else:
                output[torch.arange(len(label)).long(),
                       label] += self.kappa
            pred = torch.argmax(output, dim=1)
        else:
            pred = output
            if pred == -1:
                return pred.new_zeros(pred.shape).byte()

        return self._is_successful(pred, label)

    def update_if_smaller_dist_succeed(self, adv_img, labels, output, cost, bestl1, bestscore, o_bestl1, o_bestscore, final_adv_images):
        # images_clone.data, labels, output, cost, bestl1, bestscore, o_bestl1, o_bestscore, final_adv_images

        target_label = labels
        output_logits = output
        output_label = torch.argmax(output_logits, 1)

        mask = (cost < bestl1) & self.is_successful(output_logits, target_label, True)  # nopep8

        bestl1[mask] = cost[mask]  # redundant
        bestscore[mask] = output_label[mask]

        mask = (cost < o_bestl1) & self.is_successful(
            output_logits, target_label, True)
        o_bestl1[mask] = cost[mask]
        o_bestscore[mask] = output_label[mask]
        final_adv_images[mask] = adv_img[mask]

    def update_loss_coeffs(self, labels, bestscore, batch_size, const, upper_bound, lower_bound):
        # labels, bestscore, batch_size, const, upper_bound, lower_bound
        for i in range(batch_size):
            bestscore[i] = int(bestscore[i])
            if self.is_successful(bestscore[i], labels[i], False):
                upper_bound[i] = min(
                    upper_bound[i], const[i])

                if upper_bound[i] < 1e9:
                    const[i] = (
                        lower_bound[i] + upper_bound[i]) / 2
            else:
                lower_bound[i] = max(
                    lower_bound[i], const[i])
                if upper_bound[i] < 1e9:
                    const[i] = (
                        lower_bound[i] + upper_bound[i]) / 2
                else:
                    const[i] *= 10

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        self._check_inputs(images)

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            labels = self.get_target_label(images, labels)

        outputs = self.get_logits(images)
        # num_classes = outputs.shape[1]

        batch_size = images.shape[0]
        # coeff_lower_bound = images.new_zeros(batch_size)
        lower_bound = torch.zeros(batch_size, device=self.device)
        const = torch.ones(batch_size, device=self.device) * self.initial_const
        # coeff_upper_bound = images.new_ones(batch_size) * 1e10
        upper_bound = torch.ones(batch_size, device=self.device) * 1e10

        final_adv_images = images.clone()
        # y_one_hot = torch.zeros((labels.shape[0], num_classes), device=self.device).scatter_(1, labels, 1).float()  # nopep8
        y_one_hot = torch.eye(outputs.shape[1]).to(self.device)[labels]

        o_bestl1 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestl1 = torch.Tensor(o_bestl1).float().to(self.device)
        o_bestscore = torch.Tensor(o_bestscore).long().to(self.device)

        # Initialization: x^{(0)} = y^{(0)} = x_0 in paper Algorithm 1 part
        x_k = images.clone().detach()
        y_k = nn.Parameter(images)

        # Start binary search
        for outer_step in range(self.binary_search_steps):

            self.global_step = 0

            bestl1 = [1e10] * batch_size
            bestscore = [-1] * batch_size

            bestl1 = torch.Tensor(bestl1).float().to(self.device)
            bestscore = torch.Tensor(bestscore).long().to(self.device)

            prevloss = 1e6

            if (self.repeat and outer_step == (self.binary_search_steps - 1)):
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
                cost.backward(retain_graph=True)

                # Gradient step
                # images_parameter.data.add_(-lr, images_parameter.grad.data)
                self.global_step += 1
                with torch.no_grad():
                    y_k -= lr

                # Ploynomial decay of learning rate
                lr = self.lr * (1 - self.global_step / self.max_iterations)**0.5  # nopep8
                y_k, x_k = self.fast_ISTA(images, y_k, x_k)
                # loss ElasticNet or L1 over images_clone
                with torch.no_grad():
                    output = self.get_logits(x_k)
                    L2_loss = self.L2_loss(x_k, images)
                    L1_loss = self.L1_loss(x_k, images)
                    loss = self.EAD_loss(output, y_one_hot, L1_loss, L2_loss, const)  # nopep8

                    # print('loss: {}, prevloss: {}'.format(loss, prevloss))
                    if self.abort_early and iteration % (self.max_iterations // 10) == 0:
                        if loss > prevloss * 0.999999:
                            break
                        prevloss = loss

                    # L1 attack key step!
                    cost = L1_loss
                    self.update_if_smaller_dist_succeed(
                        x_k.data, labels, output, cost, bestl1, bestscore, o_bestl1, o_bestscore, final_adv_images)

            self.update_loss_coeffs(
                labels, bestscore, batch_size, const, upper_bound, lower_bound)

        return final_adv_images
