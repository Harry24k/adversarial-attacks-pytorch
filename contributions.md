# Guidelines for Contributors

First of all, thank you for participating in this project :heart_eyes:.
The goal of this project is to make _paper-oriented_ and _clear_ implementation that is easy to understand even for those who are new to adversarial attacks.
Thus, I note some guidelines so that contributors can easily create new attacks, so please refer to it.

## Add a New Attack

### Fork the repo

Please use git or github desktop to fork torchattacks.

### Class Inheritance

Torchattacks uses `torchattacks.Attack`.

`Attack` is the mother class of all attacks in torchattacks.

In this class, there are some functions to control the training mode of the model, the untargeted/targeted mode of the attack, the return type of adversarial images, and save the adversarial images as below:

```python
class Attack(object):
    #~~~~~#
    def __init__(self, name, model):
        r"""
        Initializes internal attack state.

        Arguments:
            name (str) : name of an attack.
            model (torch.nn.Module): model to attack.
        """

        self.attack = name
        self.model = model
        self.model_name = str(model).split("(")[0]

        self.training = model.training
        self.device = next(model.parameters()).device
        
        self._targeted = 1
        self._attack_mode = 'default'
        self._return_type = 'float'
        self._target_map_function = lambda images, labels:labels
    #~~~~~#
    def set_attack_mode(self, mode, target_map_function=None):
    #~~~~~#
    def set_return_type(self, type):
    #~~~~~#
    def save(self, data_loader, save_path=None, verbose=True):
```

The most important thing is that `Attack` only takes `model` when it is called.

Now, all methods are made on assumption that "Users feed the original images and labels". 

However, it can be changed in the future if there is a new attack that uses other inputs.
**Any ideas to further improve `Attack` class are welcome!!!**

### Define a New Attack

#### 1. Name a new method.
It is free how to name it, but if there is a common name, please use that name.

#### 2. Make a file in torchattacks/attacks
Now, let's make a file with '[name].py' in './attacks/'.

Here, **the file name must be written in lowercase** following PEP8.

For example, fgsm.py or pgd.py.

#### 3. Fill the file.
There are two things to prepare:
       1. The original paper.
       2. Attack algorithm.

As an example, here is the code of `torchattacks.PGD`:

```python
import torch
import torch.nn as nn

from ..attack import Attack


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
    
    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFALUT: 0.3)
        alpha (float): step size. (DEFALUT: 2/255)
        steps (int): number of steps. (DEFALUT: 40)
        random_start (bool): using random initialization of delta. (DEFAULT: False)
        
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=False)
        >>> adv_images = attack(images, labels)
        
    """
    def __init__(self, model, eps=0.3, alpha=2/255, steps=40, random_start=False):
        super(PGD, self).__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels = self._transform_label(images, labels)
        
        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            cost = self._targeted*loss(outputs, labels)

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

```

As above, the paper information is noted in the first line after the class definition.
Likewise here, **the class name must be written in uppercase** following PEP8.
I think it will be easy to fix the rest part of code based on other attack implementations. (This is how I do it :blush:).

#### 4. Git Pull

Finally, **PULL** your new branch to Github!!!


## TODO List

Here, I note **TODO List** for those who are interested in this.

* **Fix DeepFool**
    * Use BATCHES instead of Iterations.
    * Memory Error when calculate jacobian gradients.
    
* **Add JSMA**
    * https://github.com/Harry24k/adversarial-attacks-pytorch/issues/9
