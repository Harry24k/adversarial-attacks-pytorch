import torch
import torch.nn as nn

from ..attack import Attack


class DeepFool(Attack):
    r"""
    'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
    [https://arxiv.org/abs/1511.04599]

    Distance Measure : L2
    
    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (DEFALUT: 3)
        
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.DeepFool(model, steps=3)
        >>> adv_images = attack(images, labels)
        
    """
    def __init__(self, model, steps=3):
        super(DeepFool, self).__init__("DeepFool", model)
        self.steps = steps
        self._attack_mode = 'only_default'


    def forward(self, images, labels):
        r"""
        Overridden.
        """
        adv_images = images.clone().detach().to(self.device)

        # TODO: Use BATCHES instead of Iterations.
        for b in range(images.shape[0]):
            image = images[b:b+1, :, :, :].clone().detach().to(self.device)
            label = labels[b:b+1].clone().detach().to(self.device)
            
            for i in range(self.steps):
                image.requires_grad = True
                fs = self.model(image)[0]
                _, pre = torch.max(fs, 0)
                
                # Stop Iteration if the prediction is wrong
                if pre.item() != label.item():
                    image = torch.clamp(image, min=0, max=1).detach()
                    break
                else:
                    wrong_classes = list(range(len(fs)))
                    del wrong_classes[label.item()]
                
                ws = self.construct_jacobian(fs, image)
                
                f_0 = fs[label]
                w_0 = ws[label]
                
                f_k = fs[wrong_classes]
                w_k = ws[wrong_classes]
                
                f_prime = f_k - f_0
                w_prime = w_k - w_0
                value = torch.abs(f_prime)\
                        /torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
                _, hat_L = torch.min(value, 0)
                
                r = (torch.abs(f_prime[hat_L])\
                     /(torch.norm(w_prime[hat_L], p=2)**2))*w_prime[hat_L]

                image = torch.clamp(image + r, min=0, max=1).detach()

            adv_images[b:b+1, :, :, :] = image

        return adv_images
    
    # https://stackoverflow.com/questions/63096122/pytorch-is-it-possible-to-differentiate-a-matrix
    # torch.autograd.functional.jacobian is only for torch >= 1.5.1
    def construct_jacobian(self, y, x, retain_graph=False):
        x_grads = []
        for idx, y_element in enumerate(y.flatten()):
            if x.grad is not None:
                x.grad.zero_()
            # if specified set retain_graph=False on last iteration to clean up
            y_element.backward(retain_graph=retain_graph or idx < y.numel() - 1)
            x_grads.append(x.grad.clone().detach())
        return torch.stack(x_grads).reshape(*y.shape, *x.shape)
