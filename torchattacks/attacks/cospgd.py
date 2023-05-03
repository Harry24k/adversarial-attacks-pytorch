import torch
import torch.nn as nn

from ..attack import Attack


class CosPGD(Attack):
    r"""
    CosPGD in the paper 'CosPGD: a unified white-box adversarial attack for pixel-wise prediction tasks'
    [https://arxiv.org/abs/2302.02213]

    Distance Measure : Linf
    Scales the loss over classes based on cosine similarity to target

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.CosPGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=2255, alpha=10/255, steps=10, random_start=True):
        super().__init__("CosPGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        #one_hot_target = nn.functional.one_hot(labels).to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)
        #ne_hot_target = nn.functional.one_hot(target_labels).to(self.device)
            

        loss = nn.CrossEntropyLoss(reduction='none')
        
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + \
                torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)         
            sig_output = nn.functional.sigmoid(outputs) 
            #import ipdb;ipdb.set_trace()       
            num_classes  = outputs.shape[1]  
            if self.targeted:
                one_hot_target = nn.functional.one_hot(target_labels, num_classes=num_classes).to(self.device)
            else:
                one_hot_target = nn.functional.one_hot(labels, num_classes=num_classes).to(self.device)
            cossim = nn.functional.cosine_similarity(sig_output, one_hot_target).detach()

            # Calculate loss
            if self.targeted:                
                cost = -loss(outputs, target_labels)
                cost = (1-cossim.detach())*cost
            else:                
                cost = loss(outputs, labels)
                cost = cossim.detach()*cost

            # Update adversarial images
            cost = cost.mean()
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images,
                                min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
