import os
import copy
import itertools
import torch
import torch.nn as nn
from random import shuffle, sample

from ..attack import Attack
from ..attacks.bim import BIM

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
        trainloader (torch.utils.data.DataLoader): data loader of the unnormalized train set. Must load data in [0, 1].
        Be aware that the batch size may impact success rate. The original paper uses a batch size of 256. A different
        batch-size might require to tune the learning rate.
        lr (float): constant learning rate to collect models. In the paper, 0.05 is best for ResNet-50. 0.1 seems best
        for some other architectures. (Default: 0.05)
        epochs (int): number of epochs. (Default: 10)
        nb_models_epoch (int): number of models to collect per epoch. (Default: 4)
        wd (float): weight decay of SGD to collect models. (Default: 1e-4)
        n_grad (int): number of models to ensemble at each attack iteration. 1 (default) is recommended for efficient
        iterative attacks. Higher numbers give generally better results at the expense of computations. -1 uses all
        models (should be used for single-step attacks like FGSM).
        verbose (bool): print progress. Install the tqdm package for better print. (Default: True)

    .. note:: If a list of models is not provided to `load_models()`, the attack will start by collecting models along
    the SGD trajectory for `epochs` epochs with the constant learning rate `lr`.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height`
        and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.LGV(model, trainloader, lr=0.05, epochs=10, nb_models_epoch=4, wd=1e-4, n_grad=1, attack_class=BIM, eps=4/255, alpha=4/255/10, steps=50, verbose=True)
        >>> attack.collect_models()
        >>> attack.save_models('./models/lgv/')
        >>> adv_images = attack(images, labels)
    """

    def __init__(
        self,
        model,
        trainloader,
        lr=0.05,
        epochs=10,
        nb_models_epoch=4,
        wd=1e-4,
        n_grad=1,
        verbose=True,
        attack_class=BIM,
        **kwargs,
    ):
        model = copy.deepcopy(model)  # deep copy the model to train it
        super().__init__("LGV", model)
        self.trainloader = trainloader
        self.lr = lr
        self.epochs = epochs
        self.nb_models_epoch = nb_models_epoch
        self.wd = wd
        self.n_grad = n_grad
        self.order = "shuffle"
        self.attack_class = attack_class
        self.verbose = verbose
        self.kwargs_att = kwargs
        if not isinstance(lr, float) or lr < 0:
            raise ValueError("lr should be a non-negative float")
        if not isinstance(epochs, int) or epochs < 0:
            raise ValueError("epochs should be a non-negative integer")
        if not isinstance(nb_models_epoch, int) or nb_models_epoch < 0:
            raise ValueError("nb_models_epoch should be a non-negative integer")
        self.supported_mode = ["default", "targeted"]
        self.list_models = []
        self.base_attack = None  # will be initialized after model collection

    def collect_models(self):
        """
        Collect LGV models along the SGD trajectory
        """
        given_training = self.model.training
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.wd
        )
        loss_fn = nn.CrossEntropyLoss()
        epoch_frac = 1.0 / self.nb_models_epoch
        n_batches = int(len(self.trainloader) * epoch_frac)
        for i_sample in tqdm(
            range(self.epochs * self.nb_models_epoch), "Collecting models"
        ):
            loader = itertools.islice(self.trainloader, n_batches)
            for j, (input, target) in enumerate(loader):
                if torch.cuda.is_available():
                    input = input.to("cuda", non_blocking=True)
                    target = target.to("cuda", non_blocking=True)
                pred = self.get_logits(input)
                loss = loss_fn(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model_sample = copy.deepcopy(self.model)
            if not given_training:
                model_sample.eval()
            self.list_models.append(model_sample)
        if not given_training:
            self.model.eval()

    def load_models(self, list_models):
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
            raise RuntimeError("Call collect_models() before saving collected models.")
        os.makedirs(path, exist_ok=True)
        for i, model in enumerate(self.list_models):
            path_i = os.path.join(path, f"lgv_model_{i:05}.pt")
            torch.save({"state_dict": model.state_dict()}, path_i)

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
                print(
                    f"Phase 2: craft adversarial examples with {self.attack_class.__name__}"
                )
            self.list_models = [model.to(self.device) for model in self.list_models]
            f_model = LightEnsemble(
                self.list_models, order=self.order, n_grad=self.n_grad
            )
            if self._model_training:
                f_model.eval()
            self.base_attack = self.attack_class(
                model=f_model.to(self.device), **self.kwargs_att
            )
        # set_model_training_mode() to base attack
        self.base_attack.set_model_training_mode(
            model_training=self._model_training,
            batchnorm_training=self._batchnorm_training,
            dropout_training=self._dropout_training,
        )
        # set targeted to base attack
        if self.targeted:
            if self.attack_mode == "targeted":
                self.base_attack.set_mode_targeted_by_function(
                    target_map_function=self._target_map_function
                )
            elif self.attack_mode == "targeted(least-likely)":
                self.base_attack.set_mode_targeted_least_likely(kth_min=self._kth_min)
            elif self.attack_mode == "targeted(random)":
                self.base_attack.set_mode_targeted_random()
            else:
                raise NotImplementedError("Targeted attack mode not supported by LGV.")
        # set return type to base attack
        # self.base_attack.set_return_type(self.return_type)

        adv_images = self.base_attack(images, labels)
        return adv_images


class LightEnsemble(nn.Module):
    def __init__(self, list_models, order="shuffle", n_grad=1):
        """
        Perform a single forward pass to one of the models when call forward()

        Arguments:
            list_models (list of nn.Module): list of LGV models.
            order (str): 'shuffle' draw a model without replacement (default), 'random' draw a model with replacement,
            None cycle in provided order.
            n_grad (int): number of models to ensemble in each forward pass (fused logits). Select models according to
            `order`. If equal to -1, use all models and order is ignored.
        """
        super(LightEnsemble, self).__init__()
        self.n_models = len(list_models)
        if self.n_models < 1:
            raise ValueError("Empty list of models")
        if not (n_grad > 0 or n_grad == -1):
            raise ValueError("n_grad should be strictly positive or equal to -1")
        if order == "shuffle":
            shuffle(list_models)
        elif order in [None, "random"]:
            pass
        else:
            raise ValueError("Not supported order")
        self.models = nn.ModuleList(list_models)
        self.order = order
        self.n_grad = n_grad
        self.f_count = 0

    def forward(self, x):
        if self.n_grad >= self.n_models or self.n_grad < 0:
            indexes = list(range(self.n_models))
        elif self.order == "random":
            indexes = sample(range(self.n_models), self.n_grad)
        else:
            indexes = [
                i % self.n_models
                for i in list(range(self.f_count, self.f_count + self.n_grad))
            ]
            self.f_count += self.n_grad
        if self.n_grad == 1:
            x = self.models[indexes[0]](x)
        else:
            # clone to make sure x is not changed by inplace methods
            x_list = [
                model(x.clone()) for i, model in enumerate(self.models) if i in indexes
            ]
            x = torch.stack(x_list)
            x = torch.mean(x, dim=0, keepdim=False)
        return x
