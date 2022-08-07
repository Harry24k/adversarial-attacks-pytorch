import os
import copy
import itertools
import torch
import torch.nn as nn
from random import randrange, shuffle

from ..attack import Attack
from .bim import BIM
# fail-safe import of tqdm
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator, *args, **kwargs):
        return iterator


class LGV(Attack):
    r"""
    LGV attack in the paper 'LGV: Boosting Adversarial Example Transferability from Large Geometric Vicinity'
    [https://arxiv.org/abs/2207.13129]

    Arguments:
        model (nn.Module): initial model to attack.
        trainloader (torch.utils.data.DataLoader): data loader of the unnormalized train set. Must load data in [0, 1]
        without normalization to be fed to the model.
        lr (float): constant learning rate to collect models. In the paper, 0.05 is best for ResNet-50. 0.1 seems best
        for some other architectures. (Default: 0.05)
        epochs (int): number of epochs. (Default: 10)
        nb_models_epoch (int): number of models to collect per epoch. (Default: 4)
        wd (float): weight decay of SGD to collect models. (Default: 4)
        full_grad (bool): If False, every gradient is of a single collected model randomly picked without replacement
        (recommended for efficient iterative attack). If True, every gradient is of all models (should be used for
        single step attacks like FGSM). (Default: False)
        verbose (bool): print progress. Install tqdm package for better print. (Default: True)

    .. note:: If a list of models is not provided to `load_models()`, the attack will start by collecting models along
    the SGD trajectory for `epochs` epochs with the constant learning rate `lr`.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.LGV(model, trainloader, lr=0.05, epochs=10, nb_models_epoch=4, wd=1e-4, full_grad=False, attack_class=BIM, eps=4/255, alpha=4/255/10, steps=50, verbose=True)
        >>> attack.collect_models()
        >>> attack.save_collected_models('./lgv_models/')
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, trainloader, lr=0.05, epochs=10, nb_models_epoch=4, wd=1e-4, full_grad=False,
                 verbose=True, attack_class=BIM, **kwargs):
        model = copy.deepcopy(model)  # deep copy the model to train it
        super().__init__("LGV", model)
        self.trainloader = trainloader
        self.lr = lr
        self.epochs = epochs
        self.nb_models_epoch = nb_models_epoch
        self.wd = wd
        self.full_grad = full_grad
        self.attack_class = attack_class
        self.verbose = verbose
        self.kwargs_att = kwargs
        if not isinstance(lr, float) or lr < 0:
            raise ValueError('lr should be a non-negative float')
        if not isinstance(epochs, int) or epochs < 0:
            raise ValueError('epochs should be a non-negative integer')
        if not isinstance(nb_models_epoch, int) or nb_models_epoch < 0:
            raise ValueError('nb_models_epoch should be a non-negative integer')
        self._supported_mode = ['default', 'targeted']
        self.list_models = []
        self.base_attack = None  # will be initialized after model collection

    def collect_models(self):
        """
        Collect LGV models along the SGD trajectory
        """
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.wd
        )
        loss_fn = nn.CrossEntropyLoss()
        epoch_frac = 1.0 / self.nb_models_epoch
        n_batches = int(len(self.trainloader) * epoch_frac)
        for i_sample in tqdm(range(self.epochs * self.nb_models_epoch), 'Collecting models'):
            loader = itertools.islice(self.trainloader, n_batches)
            for j, (input, target) in enumerate(loader):
                if torch.cuda.is_available():
                    input = input.to('cuda', non_blocking=True)
                    target = target.to('cuda', non_blocking=True)
                pred = self.model(input)
                loss = loss_fn(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.list_models.append(copy.deepcopy(self.model))

    def load_collected_models(self, list_models):
        """
        Load collected models

        Arguments:
        list_models (list of nn.Module): list of LGV models.
        """
        if not isinstance(list_models, list):
            raise ValueError("list_models should be a list of pytorch models")
        self.list_models = list_models

    def save_models(self, path):
        """
        Save collected models to the `path` directory

        Arguments:
        path (str): directory where to save models.
        """
        if len(self.list_models) == 0:
            raise RuntimeError('Call collect_models() before saving collected models.')
        os.makedirs(path, exist_ok=True)
        for i, model in enumerate(self.list_models):
            path_i = os.path.join(path, f'lgv_model_{i:05}.pt')
            torch.save({'state_dict': model.state_dict()}, path_i)

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        if len(self.list_models) == 0:
            if self.verbose:
                print(f"Phase 1: collect models for {self.epochs} epochs")
            self.collect_models()

        if not self.base_attack:
            if self.verbose:
                print(f"Phase 2: craft adversarial examples with {self.attack_class.__name__}")
            # TODO: might not support set_training_mode() correctly:
            self.list_models = [model.eval() for model in self.list_models]
            f_model = LightEnsemble(self.list_models, order='shuffle', full_grad=self.full_grad)
            # TODO: add support for targeted attacks
            self.base_attack = self.attack_class(model=f_model, **self.kwargs_att)

        adv_images = self.base_attack(images, labels)
        return adv_images


class LightEnsemble(nn.Module):

    def __init__(self, list_models, order='shuffle', full_grad=False):
        """
        Perform a single forward pass to one of the models when call forward()

        Arguments:
            list_models (list of nn.Module): list of LGV models.
            order: str, 'shuffle' draw a model without replacement (default), 'random' draw a model with replacement,
            None cycle in provided order.
            full_grad (bool): if True, a forward pass is performed on all models and logits are fused and order is
            ignored. If False, a forward pass is performed on a single model chosen according to `order` (default).
        """
        super(LightEnsemble, self).__init__()
        self.n_models = len(list_models)
        if self.n_models < 1:
            raise ValueError('Empty list of models')
        if order == 'shuffle':
            shuffle(list_models)
        elif order in [None, 'random']:
            pass
        else:
            raise ValueError('Not supported order')
        self.models = nn.ModuleList(list_models)
        self.order = order
        self.full_grad = full_grad
        self.f_count = 0

    def forward(self, x):
        if self.full_grad:
            # clone to make sure x is not changed by inplace methods
            x_list = [model(x.clone()) for model in self.models]
            x = torch.stack(x_list)
            x = torch.mean(x, dim=0, keepdim=False)
        else:
            if self.order == 'random':
                index = randrange(0, self.n_models)
            else:
                index = self.f_count % self.n_models
            x = self.models[index](x)
        self.f_count += 1
        return x
