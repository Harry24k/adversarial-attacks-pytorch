import torch
import torch.nn as nn

from ..attack import Attack


class PatchFool(Attack):
    r"""
    PatchFoolAttack in the paper 'PATCH-FOOL: ARE VISION TRANSFORMERS ALWAYS ROBUST AGAINST ADVERSARIAL PERTURBATIONS?'
    [https://arxiv.org/pdf/2203.08392]

    Inspired by https://github.com/GATECH-EIC/Patch-Fool

    Distance Measure : Linf, L2, number of patches to change

    Arguments:
        ViT model (nn.Module): model to attack, must return logits as well as attentions: List[Tensor]
        patch_size (int): ViT patch size
        patch_select (Option[Rand, Saliency, Attn]): mode of patch selecting for attack
        num_patch: number of patches to attack
        atten_select: attention layer id to attack
        mild_l_inf: constraints in linf norm
        mild_l_2: constraints in l2 norm
        random_sparse_pixel: choose atacked pixels randomly
        attack_learning_rate: lr
        step_size: scheduler step size
        gamma: scheduler gamma
        train_attack_iters: lenth of atack
        learnable_mask_stop: lenth of mask learning
        attack_mode: attack mode
        atten_loss_weight: weight of attention component in loss
        mu: dataset specific mean
        std: dataset specific std

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::

    """

    def __init__(self, 
                 model, 
                 patch_size, 
                 patch_select = "Attn", 
                 num_patch = 1,
                 atten_select = 4, 
                 sparse_pixel_num = 0,
                 mild_l_inf = 0,
                 mild_l_2 = 0,
                 random_sparse_pixel = False,
                 attack_learning_rate = 0.22,
                 step_size = 10, 
                 gamma = 0.95,
                 train_attack_iters = 250,
                 learnable_mask_stop = 200,
                 attack_mode = "CE_loss",
                 atten_loss_weight = 0.002,
                 mu = torch.tensor([0.4914, 0.4822, 0.4465]).unsqueeze(-1).unsqueeze(-1),
                 std = torch.tensor([0.2471, 0.2435, 0.2616]).unsqueeze(-1).unsqueeze(-1)
                 ):
        super().__init__("PatchFool", model)
        self.patch_size = patch_size
        self.num_patch = num_patch

        if patch_select in ["Rand", "Saliency", "Attn"]:
            self.patch_select = patch_select
        else:
            print(f'Unknown patch_select: {patch_select}\nSelect one from [Rand, Saliency, Attn]')
            raise

        self.atten_select = atten_select
        self.sparse_pixel_num = sparse_pixel_num

        self.mild_l_inf = mild_l_inf
        self.mild_l_2 = mild_l_2
        
        self.mu = mu.to(self.device)
        self.std = std.to(self.device)

        self.random_sparse_pixel = random_sparse_pixel
        self.attack_learning_rate = attack_learning_rate

        self.gamma = gamma
        self.step_size = step_size

        self.train_attack_iters = train_attack_iters
        self.learnable_mask_stop = learnable_mask_stop
        self.atten_loss_weight = atten_loss_weight

        if attack_mode in ['CE_loss', 'Attention']:
            self.attack_mode = attack_mode
        else:
            print(f'Unknown patch_select: {attack_mode}\nSelect one from [CE_loss, Attention]')
            raise 

    def get_logits_and_attention(self, inputs, labels=None, *args, **kwargs):
        if self._normalization_applied is False:
            inputs = self.normalize(inputs)
        logits, atten = self.model(inputs, return_attn=True)
        return logits, atten

    @staticmethod
    def PCGrad(atten_grad, ce_grad, sim, shape):
        pcgrad = atten_grad[sim < 0]
        temp_ce_grad = ce_grad[sim < 0]
        dot_prod = torch.mul(pcgrad, temp_ce_grad).sum(dim=-1)
        dot_prod = dot_prod / torch.norm(temp_ce_grad, dim=-1)
        pcgrad = pcgrad - dot_prod.view(-1, 1) * temp_ce_grad
        atten_grad[sim < 0] = pcgrad
        atten_grad = atten_grad.view(shape)
        return atten_grad

    @staticmethod
    def clamp(X, lower_limit, upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        criterion = nn.CrossEntropyLoss()

        patch_num_per_line = int(images.shape[-1] / self.patch_size)
        delta = torch.zeros_like(images).to(self.device)
        delta.requires_grad = True

        self.model.zero_grad()
        
        out, atten = self.get_logits_and_attention(images + delta)

        loss = criterion(out, labels)

        '''choose patch'''
        # max_patch_index size: [Batch, num_patch attack]
        if self.patch_select == 'Rand':
            '''random choose patch'''
            max_patch_index = torch.randint(low = 0, high = patch_num_per_line * patch_num_per_line, size = (images.shape[0], self.num_patch))
        elif self.patch_select == 'Saliency':
            '''gradient based method'''
            grad = torch.autograd.grad(loss, delta)[0]
            grad = torch.abs(grad)
            filter = torch.ones([1, 3, self.patch_size, self.patch_size]).float().to(self.device)
            patch_grad = nn.functional.conv2d(grad, filter, stride=self.patch_size)
            patch_grad = patch_grad.view(patch_grad.shape[0], -1)
            max_patch_index = patch_grad.argsort(descending=True)[:, :self.num_patch]
        elif self.patch_select == 'Attn':
            '''attention based method'''
            atten_layer = atten[self.atten_select].mean(dim=1)
            atten_layer = atten_layer.mean(dim=-2)
            max_patch_index = atten_layer.argsort(descending=True)[:, :self.num_patch] #choose num_patch patches to atttack 

        '''build mask'''
        mask = torch.zeros(images.shape[0], 1, images.shape[2], images.shape[3]).to(self.device)
        if self.sparse_pixel_num != 0:
            learnable_mask = mask.clone()

        for j in range(images.shape[0]):
            index_list = max_patch_index[j]
            for index in index_list:
                row = (index // patch_num_per_line) * self.patch_size
                column = (index % patch_num_per_line) * self.patch_size

                if self.sparse_pixel_num != 0:
                    learnable_mask.data[j, :, row:row + self.patch_size, column:column + self.patch_size] = torch.rand(
                        [self.patch_size, self.patch_size])
                mask[j, :, row:row + self.patch_size, column:column + self.patch_size] = 1


        '''adv attack'''
        max_patch_index_matrix = max_patch_index[:, 0]
        max_patch_index_matrix = max_patch_index_matrix.repeat(atten[0].shape[-1], 1)
        max_patch_index_matrix = max_patch_index_matrix.permute(1, 0)
        max_patch_index_matrix = max_patch_index_matrix.flatten().long()

        if self.mild_l_inf == 0:
            '''random init delta'''
            delta = (torch.rand_like(images) - self.mu) / self.std
        else:
            '''constrain delta: range [x-epsilon, x+epsilon]'''
            epsilon = self.mild_l_inf / self.std
            delta = 2 * epsilon * torch.rand_like(images) - epsilon + images

        delta.data = PatchFool.clamp(delta, (0 - self.mu) / self.std, (1 - self.mu) / self.std)
        original_img = images.clone()

        if self.random_sparse_pixel:
            '''random select pixels'''
            sparse_mask = torch.zeros_like(mask)
            learnable_mask_temp = learnable_mask.view(learnable_mask.shape[0], -1)
            sparse_mask_temp = sparse_mask.view(sparse_mask.shape[0], -1)
            value, _ = learnable_mask_temp.sort(descending=True)
            threshold = value[:, self.sparse_pixel_num - 1].view(-1, 1)
            sparse_mask_temp[learnable_mask_temp >= threshold] = 1
            mask = sparse_mask

        if self.sparse_pixel_num == 0 or self.random_sparse_pixel:
            images = torch.mul(images, 1 - mask)
        else:
            '''select by learnable mask'''
            learnable_mask.requires_grad = True
        delta = delta.to(self.device)
        delta.requires_grad = True

        opt = torch.optim.Adam([delta], lr=self.attack_learning_rate)
        if self.sparse_pixel_num != 0 and (not self.random_sparse_pixel):
            mask_opt = torch.optim.Adam([learnable_mask], lr=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=self.step_size, gamma=self.gamma)

        '''Start Adv Attack'''
        for train_iter_num in range(self.train_attack_iters):
            self.model.zero_grad()
            opt.zero_grad()

            '''Build Sparse Patch attack binary mask'''
            if self.sparse_pixel_num != 0 and (not self.random_sparse_pixel):
                if train_iter_num < self.learnable_mask_stop:
                    mask_opt.zero_grad()
                    sparse_mask = torch.zeros_like(mask)
                    learnable_mask_temp = learnable_mask.view(learnable_mask.shape[0], -1)
                    sparse_mask_temp = sparse_mask.view(sparse_mask.shape[0], -1)
                    value, _ = learnable_mask_temp.sort(descending=True)

                    threshold = value[:, self.sparse_pixel_num-1].view(-1, 1)
                    sparse_mask_temp[learnable_mask_temp >= threshold] = 1

                    '''inference as sparse_mask but backward as learnable_mask'''
                    temp_mask = ((sparse_mask - learnable_mask).detach() + learnable_mask) * mask
                else:
                    temp_mask = sparse_mask

                images = original_img * (1-sparse_mask)
                out, atten = self.get_logits_and_attention(images + torch.mul(delta, temp_mask))

            else:
                out, atten = self.get_logits_and_attention(images + torch.mul(delta, mask))

            '''final CE-loss'''
            loss = criterion(out, labels)

            if self.attack_mode == 'Attention':
                grad = torch.autograd.grad(loss, delta, retain_graph=True)[0]
                ce_loss_grad_temp = grad.view(images.shape[0], -1).detach().clone()
                if self.sparse_pixel_num != 0 and (not self.random_sparse_pixel) and train_iter_num < self.learnable_mask_stop:
                    mask_grad = torch.autograd.grad(loss, learnable_mask, retain_graph=True)[0]

                # Attack the first 6 layers' Attn
                range_list = range(len(atten)//2)
                for atten_num in range_list:
                    if atten_num == 0:
                        continue
                    atten_map = atten[atten_num]
                    atten_map = atten_map.mean(dim=1)
                    atten_map = atten_map.view(-1, atten_map.shape[-1])
                    atten_map = -torch.log(atten_map)
                    atten_loss = nn.functional.nll_loss(atten_map, max_patch_index_matrix)

                    atten_grad = torch.autograd.grad(atten_loss, delta, retain_graph=True)[0]

                    atten_grad_temp = atten_grad.view(images.shape[0], -1)
                    cos_sim = nn.functional.cosine_similarity(atten_grad_temp, ce_loss_grad_temp, dim=1)

                    if self.sparse_pixel_num != 0 and (not self.random_sparse_pixel) and train_iter_num < self.learnable_mask_stop:
                        mask_atten_grad = torch.autograd.grad(atten_loss, learnable_mask, retain_graph=True)[0]

                    '''PCGrad'''
                    atten_grad = PatchFool.PCGrad(atten_grad_temp, ce_loss_grad_temp, cos_sim, grad.shape)
                    if self.sparse_pixel_num != 0 and (not self.random_sparse_pixel):
                        mask_atten_grad_temp = mask_atten_grad.view(mask_atten_grad.shape[0], -1)
                        ce_mask_grad_temp = mask_grad.view(mask_grad.shape[0], -1)
                        mask_cos_sim = nn.functional.cosine_similarity(mask_atten_grad_temp, ce_mask_grad_temp, dim=1)
                        mask_atten_grad = PatchFool.PCGrad(mask_atten_grad_temp, ce_mask_grad_temp, mask_cos_sim, mask_atten_grad.shape)

                    grad += atten_grad * self.atten_loss_weight
                    if self.sparse_pixel_num != 0 and (not self.random_sparse_pixel):
                        mask_grad += mask_atten_grad * self.atten_loss_weight

            else:
                '''no attention loss'''
                if self.sparse_pixel_num != 0 and (not self.random_sparse_pixel) and train_iter_num < self.learnable_mask_stop:
                    grad = torch.autograd.grad(loss, delta, retain_graph=True)[0]
                    mask_grad = torch.autograd.grad(loss, learnable_mask)[0]
                else:
                    grad = torch.autograd.grad(loss, delta)[0]

            opt.zero_grad()
            delta.grad = -grad
            opt.step()
            scheduler.step()

            if self.sparse_pixel_num != 0 and (not self.random_sparse_pixel) and train_iter_num < self.learnable_mask_stop:
                mask_opt.zero_grad()
                learnable_mask.grad = -mask_grad
                mask_opt.step()

                learnable_mask_temp = learnable_mask.view(images.shape[0], -1)
                learnable_mask.data -= learnable_mask_temp.min(-1)[0].view(-1, 1, 1, 1)
                learnable_mask.data += 1e-6
                learnable_mask.data *= mask

            '''l2 constrain'''
            if self.mild_l_2 != 0:
                radius = (self.mild_l_2 / self.std).squeeze()
                perturbation = (delta.detach() - original_img) * mask
                l2 = torch.linalg.norm(perturbation.view(perturbation.shape[0], perturbation.shape[1], -1), dim=-1)
                radius = radius.repeat([l2.shape[0], 1])
                l2_constraint = radius / l2
                l2_constraint[l2 < radius] = 1.
                l2_constraint = l2_constraint.view(l2_constraint.shape[0], l2_constraint.shape[1], 1, 1)
                delta.data = original_img + perturbation * l2_constraint

            '''l_inf constrain'''
            if self.mild_l_inf != 0:
                epsilon = self.mild_l_inf / self.std
                delta.data = PatchFool.clamp(delta, original_img - epsilon, original_img + epsilon)

            delta.data = PatchFool.clamp(delta, (0 - self.mu) / self.std, (1 - self.mu) / self.std)

        '''Eval Adv Attack'''
        perturb_images = None
        with torch.no_grad():
            if self.sparse_pixel_num == 0 or self.random_sparse_pixel:
                perturb_images = images + torch.mul(delta, mask)
            else:
                if train_iter_num < self.learnable_mask_stop:
                    sparse_mask = torch.zeros_like(mask)
                    learnable_mask_temp = learnable_mask.view(learnable_mask.shape[0], -1)
                    temp_mask = sparse_mask.view(sparse_mask.shape[0], -1)
                    value, _ = learnable_mask_temp.sort(descending=True)
                    threshold = value[:, self.sparse_pixel_num - 1].view(-1, 1)
                    temp_mask[learnable_mask_temp >= threshold] = 1

                images = original_img * (1 - sparse_mask)
                perturb_images = images + torch.mul(delta, sparse_mask)

        return perturb_images

            

