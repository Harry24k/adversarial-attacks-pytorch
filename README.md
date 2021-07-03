# Adversarial-Attacks-PyTorch

<p>
  <a href="https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/github/license/Harry24k/adversarial-attacks-pytorch" /></a>
  <a href="https://pypi.org/project/torchattacks/"><img alt="Pypi" src="https://img.shields.io/pypi/v/torchattacks.svg" /></a>
  <a href="https://adversarial-attacks-pytorch.readthedocs.io/en/latest/"><img alt="Documentation Status" src="https://readthedocs.org/projects/adversarial-attacks-pytorch/badge/?version=latest" /></a>
</p>

[README [KOR]](https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/README_KOR.md)

[Torchattacks](https://www.semanticscholar.org/paper/Torchattacks-%3A-A-Pytorch-Repository-for-Adversarial-Kim/1f4b3283faf534ef92d7d7fa798b26480605ead9) is a PyTorch library that contains adversarial attacks to generate adversarial examples.


<p align="center">

|                         Clean Image                          |                      Adversarial Image                       |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/clean.png" width="300" height="300"> | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/pgd.png" width="300" height="300"> |

</p>

## Table of Contents
1. [Usage](#Usage)
2. [Attacks and Papers](#Attacks-and-Papers)
3. [Performance Comparison](#Performance-Comparison)
4. [Documentation](#Documentation)
5. [Citation](#Citation)
6. [Expanding the Usage](#Expanding-the-Usage)
7. [Contribution](#Contribution)
8. [Recommended Sites and Packages](#Recommended-Sites-and-Packages)



## Usage

### :clipboard: Dependencies

- torch==1.4.0
- python==3.6



### :hammer: Installation

- `pip install torchattacks` or
- `git clone https://github.com/Harry24k/adversairal-attacks-pytorch`

```python
import torchattacks
atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=4)
adversarial_images = atk(images, labels)
```



###  :warning: Precautions

* **All images should be scaled to [0, 1] with transform[to.Tensor()] before used in attacks.** To make it easy to use adversarial attacks, a reverse-normalization is not included in the attack process. To apply an input normalization, please add a normalization layer to the model. Please refer to [code](https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/White%20Box%20Attack%20(ImageNet).ipynb) or [nbviewer](https://nbviewer.jupyter.org/github/Harry24k/adversarial-attacks-pytorch/blob/master/demos/White%20Box%20Attack%20%28ImageNet%29.ipynb).

* **All models should return ONLY ONE vector of `(N, C)` where `C = number of classes`.** Considering most models in _torchvision.models_ return one vector of `(N,C)`, where `N` is the number of inputs and `C` is thenumber of classes, _torchattacks_ also only supports limited forms of output.  Please check the shape of the model’s output carefully. In the case of the model returns multiple outputs, please refer to [the demo](https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Model%20with%20Multiple%20Outputs.ipynb).

* **`torch.backends.cudnn.deterministic = True` to get same adversarial examples with fixed random seed**. Some operations are non-deterministic with float tensors on GPU [[discuss]](https://discuss.pytorch.org/t/inconsistent-gradient-values-for-the-same-input/26179). If you want to get same results with same inputs, please run `torch.backends.cudnn.deterministic = True`[[ref]](https://stackoverflow.com/questions/56354461/reproducibility-and-performance-in-pytorch).




## Attacks and Papers

Implemented adversarial attacks in the papers.

The distance measure in parentheses.

|          Name          | Paper                                                        | Remark                                                       |
| :--------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
|  **FGSM**<br />(Linf)  | Explaining and harnessing adversarial examples ([Goodfellow et al., 2014](https://arxiv.org/abs/1412.6572)) |                                                              |
|  **BIM**<br />(Linf)   | Adversarial Examples in the Physical World ([Kurakin et al., 2016](https://arxiv.org/abs/1607.02533)) | Basic iterative method or Iterative-FSGM                     |
|    **CW**<br />(L2)    | Towards Evaluating the Robustness of Neural Networks ([Carlini et al., 2016](https://arxiv.org/abs/1608.04644)) |                                                              |
| **RFGSM**<br />(Linf)  | Ensemble Adversarial Traning: Attacks and Defences ([Tramèr et al., 2017](https://arxiv.org/abs/1705.07204)) | Random initialization + FGSM                                 |
|  **PGD**<br />(Linf)   | Towards Deep Learning Models Resistant to Adversarial Attacks ([Mardry et al., 2017](https://arxiv.org/abs/1706.06083)) | Projected Gradient Method                                    |
|  **PGDL2**<br />(L2)   | Towards Deep Learning Models Resistant to Adversarial Attacks ([Mardry et al., 2017](https://arxiv.org/abs/1706.06083)) | Projected Gradient Method                                    |
| **MIFGSM**<br />(Linf) | Boosting Adversarial Attacks with Momentum ([Dong et al., 2017](https://arxiv.org/abs/1710.06081)) | :heart_eyes: Contributor [zhuangzi926](https://github.com/zhuangzi926), [huitailangyz](https://github.com/huitailangyz) |
|  **TPGD**<br />(Linf)  | Theoretically Principled Trade-off between Robustness and Accuracy ([Zhang et al., 2019](https://arxiv.org/abs/1901.08573)) |                                                              |
|  **EOTPGD**<br />(Linf)  | Comment on "Adv-BNN: Improved Adversarial Defense through Robust Bayesian Neural Network" ([Zimmermann, 2019](https://arxiv.org/abs/1907.00895)) | [EOT](https://arxiv.org/abs/1707.07397)+PGD                |
| **PGDDLR**<br />(Linf)  | Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks ([Croce et al., 2020](https://arxiv.org/abs/2001.03994)) | PGD based on DLR loss                                 |
| **APGD**<br />(Linf, L2) | Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks ([Croce et al., 2020](https://arxiv.org/abs/2001.03994)) |                                  |
| **APGDT**<br />(Linf, L2) | Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks ([Croce et al., 2020](https://arxiv.org/abs/2001.03994)) | Targeted APGD                                 |
| **FAB**<br />(Linf, L2, L1) | Minimally distorted Adversarial Examples with a Fast Adaptive Boundary Attack ([Croce et al., 2019](https://arxiv.org/abs/1907.02044)) |                               |
| **Square**<br />(Linf, L2) | Square Attack: a query-efficient black-box adversarial attack via random search ([Andriushchenko et al., 2019](https://arxiv.org/abs/1912.00049)) |                                |
| **AutoAttack**<br />(Linf, L2) | Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks ([Croce et al., 2020](https://arxiv.org/abs/2001.03994)) | APGD+APGDT+FAB+Square                                 |
| **DeepFool**<br />(L2) | DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks ([Moosavi-Dezfooli et al., 2016](https://arxiv.org/abs/1511.04599)) |                               |
| **OnePixel**<br />(L0) | One pixel attack for fooling deep neural networks ([Su et al., 2019](https://arxiv.org/abs/1710.08864)) |                                |
| **SparseFool**<br />(L0) | SparseFool: a few pixels make a big difference ([Modas et al., 2019](https://arxiv.org/abs/1811.02248)) |                                  |
| **DI2FGSM**<br />(Linf) | Improving Transferability of Adversarial Examples with Input Diversity ([Xie et al., 2019](https://arxiv.org/abs/1803.06978)) |                                  |
| **TIFGSM**<br />(Linf) | Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks ([Dong et al., 2019](https://arxiv.org/abs/1803.06978)) |                                  |



## Performance Comparison

For a fair comparison, [Robustbench](https://github.com/RobustBench/robustbench) is used. As for the comparison packages, currently updated and the most cited methods were selected:

* **Foolbox**: [178](https://scholar.google.com/scholar?q=Foolbox%3A%20A%20Python%20toolbox%20to%20benchmark%20the%20robustness%20of%20machine%20learning%20models.%20arXiv%202018) citations and last update 2020.10.19.

* **ART**: [102](https://scholar.google.com/scholar?cluster=5391305326811305758&hl=ko&as_sdt=0,5&sciodt=0,5) citations and last update 2020.12.11.

Robust accuracy against each attack and elapsed time on the first 50 images of CIFAR10. For L2 attacks, the average L2 distances between adversarial images and the original images are recorded. All experiments were done on GeForce RTX 2080. This is for `torchattacks==2.13.2`. For the latest version, please refer to here ([code](https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Performance%20Comparison%20(CIFAR10).ipynb), [nbviewer](https://nbviewer.jupyter.org/github/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Performance%20Comparison%20(CIFAR10).ipynb)).

|  **Attack (Linf)**   |     **Package**     | [Wong2020](https://arxiv.org/abs/2001.03994) | [Rice2020](https://arxiv.org/abs/2002.11569) | [Carmon2019](https://arxiv.org/abs/1905.13736) |     **Remark**     |
| :----------------: | :-----------------: | -------------------------------------------: | -------------------------------------------: | ---------------------------------------------: | :----------------: |
|      **FGSM**      |    Torchattacks     |                                 48% (15 ms) |                                 62% (94 ms) |                                  68% (18 ms) |                    |
|                    | Foolbox<sup>*</sup> |                                 48% (34 ms) |                                62% (46 ms) |                                   68% (27 ms) |                    |
|                    |         ART         |                                48% (50 ms) |                               62% (799 ms) |                                 68% (221 ms) |                    |
|      **PGD**       |    Torchattacks     |                              44% (156 ms) |                             58% (2770 ms) |                                 58% (692 ms) |                    |
|                    | Foolbox<sup>*</sup> |                                44% (249 ms) |                             58% (3452 ms) |                                58% (945 ms) |                    |
|                    |         ART         |                             44% (495 ms) |                             58% (4782 ms) |                               58% (1444 ms) |                    |
| **Attack (L2)**  |     **Package**     | [Wong2020](https://arxiv.org/abs/2001.03994) | [Rice2020](https://arxiv.org/abs/2002.11569) | [Carmon2019](https://arxiv.org/abs/1905.13736) |     **Remark**     |
| **CW<sup>†</sup>** |    Torchattacks     |               14% / 0.61 <br />(4367 ms) |            22% / 0.56<br />(43680 ms) |               26% / 0.48<br />(13032 ms) | **Highest Success Rate** |
|                    | Foolbox<sup>*</sup> |               32% / 0.41 <br />(4530 ms) |               34% / 0.43<br />(45273 ms) |                 32% / 0.42<br />(13314 ms) | **Smallest Perturbation** |
|                    |         ART         |             24% / 0.71<br />(71613 ms) |           26% / 0.65<br />(691977 ms) |               26% / 0.62<br />(206250 ms) |  |
| **PGDL2** |    Torchattacks     |                     68% / 0.5<br /> (166 ms) |                     70% / 0.5<br />(2796 ms) |                       68% / 0.5<br /> (712 ms) |                           |
|                    | Foolbox<sup>*</sup> |                     68% / 0.5<br /> (267 ms) |                    70% / 0.5<br /> (3501 ms) |                       68% / 0.5<br /> (962 ms) |                    |
|                    |         ART         |                     68% / 0.5<br /> (470 ms) |                    70% / 0.5<br /> (4822 ms) |                       68% / 0.5<br />(1441 ms) |                           |

<sup>*</sup> Note that Foolbox returns accuracy and adversarial images simultaneously, thus the *actual* time for generating adversarial images  might be shorter than the records.

<sup>**†**</sup>Considering that the binary search algorithm for const `c` can be time-consuming, torchattacks supports customized grid search as in [code](https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Applications%20of%20MultiAttack%20(CIFAR10).ipynb), [nbviewer](https://nbviewer.jupyter.org/github/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Applications%20of%20MultiAttack%20(CIFAR10).ipynb).



## Documentation

### :book: ReadTheDocs

Here is a [documentation](https://adversarial-attacks-pytorch.readthedocs.io/en/latest/index.html) for this package.



### :mag_right: Update Records

Here is [update records](update_records.md) of this package.



### :rocket: Demos

- **White Box Attack with ImageNet** ([code](https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/White%20Box%20Attack%20(ImageNet).ipynb), [nbviewer](https://nbviewer.jupyter.org/github/Harry24k/adversarial-attacks-pytorch/blob/master/demos/White%20Box%20Attack%20%28ImageNet%29.ipynb)):  Using _torchattacks_ to make adversarial examples with [the ImageNet dataset](http://www.image-net.org/) to fool ResNet-18.
- **Black Box Attack with CIFAR10** ([code](https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Black%20Box%20Attack%20(CIFAR10).ipynb), [nbviewer](https://nbviewer.jupyter.org/github/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Black%20Box%20Attack%20%28CIFAR10%29.ipynb)):  This demo provides an example of black box attack with two different models. First, make adversarial datasets from a holdout model with CIFAR10 and save it as torch dataset. Second, use the adversarial datasets to attack a target model.
- **Adversairal Training with MNIST** ([code](https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/demos/Adversairal%20Training%20(MNIST).ipynb), [nbviewer](https://nbviewer.jupyter.org/github/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Adversairal%20Training%20%28MNIST%29.ipynb)):  This code shows how to do adversarial training with this repository. The MNIST dataset and a custom model are used in this code. The adversarial training is performed with PGD, and then FGSM is applied to evaluate the model.
- **Applications of MultiAttack with CIFAR10** ([code](https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Applications%20of%20MultiAttack%20(CIFAR10).ipynb), [nbviewer](https://nbviewer.jupyter.org/github/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Applications%20of%20MultiAttack%20(CIFAR10).ipynb)):  This code shows the applications of _Multiattack_.



## Citation
[SemanticScholar](https://www.semanticscholar.org/paper/Torchattacks-%3A-A-Pytorch-Repository-for-Adversarial-Kim/1f4b3283faf534ef92d7d7fa798b26480605ead9).
If you use this package, please cite the following BibTex:

```
@article{kim2020torchattacks,
  title={Torchattacks: A Pytorch Repository for Adversarial Attacks},
  author={Kim, Hoki},
  journal={arXiv preprint arXiv:2010.01950},
  year={2020}
}
```



## Expanding the Usage

Torchattacks supports collaboration with other attack packages.

Through expending the usage, we can use fucntions in _torchattacks_ such as _save_ and _multiattack_.



###  :milky_way: FoolBox

* https://github.com/bethgelab/foolbox
* `pip install foolbox`
* e.g., L2BrendelBethge

```python
from torchattacks.attack import Attack
import foolbox as fb

class L2BrendelBethge(Attack):
    def __init__(self, model):
        super(L2BrendelBethge, self).__init__("L2BrendelBethge", model)
        self.fmodel = fb.PyTorchModel(self.model, bounds=(0,1), device=self.device)
        self.init_attack = fb.attacks.DatasetAttack()
        self.adversary = fb.attacks.L2BrendelBethgeAttack(init_attack=self.init_attack)
        self._attack_mode = 'only_default'
        
    def forward(self, images, labels):
        images, labels = images.to(self.device), labels.to(self.device)
        
        # DatasetAttack
        batch_size = len(images)
        batches = [(images[:batch_size//2], labels[:batch_size//2]),
                   (images[batch_size//2:], labels[batch_size//2:])]
        self.init_attack.feed(model=self.fmodel, inputs=batches[0][0]) # feed 1st batch of inputs
        self.init_attack.feed(model=self.fmodel, inputs=batches[1][0]) # feed 2nd batch of inputs
        criterion = fb.Misclassification(labels)
        init_advs = self.init_attack.run(self.fmodel, images, criterion)
        
        # L2BrendelBethge
        adv_images = self.adversary.run(self.fmodel, images, labels, starting_points=init_advs)
        return adv_images

atk = L2BrendelBethge(model)
atk.save(data_loader=test_loader, save_path="_temp.pt", verbose=True)
```



###  :milky_way: Adversarial-Robustness-Toolbox (ART)

* https://github.com/IBM/adversarial-robustness-toolbox
* `git clone https://github.com/IBM/adversarial-robustness-toolbox`
* e.g., SaliencyMapMethod (or Jacobian based saliency map attack)

```python
import torch.nn as nn
import torch.optim as optim

from torchattacks.attack import Attack

import art.attacks.evasion as evasion
from art.classifiers import PyTorchClassifier

class JSMA(Attack):
    def __init__(self, model, theta=1/255, gamma=0.15, batch_size=128):
        super(JSMA, self).__init__("JSMA", model)
        self.classifier = PyTorchClassifier(
                            model=self.model, clip_values=(0, 1),
                            loss=nn.CrossEntropyLoss(),
                            optimizer=optim.Adam(self.model.parameters(), lr=0.01),
                            input_shape=(1, 28, 28), nb_classes=10)
        self.adversary = evasion.SaliencyMapMethod(classifier=self.classifier,
                                                   theta=theta, gamma=gamma,
                                                   batch_size=batch_size)
        self.target_map_function = lambda labels: (labels+1)%10
        self._attack_mode = 'only_default'
        
    def forward(self, images, labels):
        adv_images = self.adversary.generate(images, self.target_map_function(labels))
        return torch.tensor(adv_images).to(self.device)

atk = JSMA(model)
atk.save(data_loader=test_loader, save_path="_temp.pt", verbose=True)
```



## Contribution

All kind of contributions are always welcome! :blush:

If you are interested in adding a new attack to this repo or fixing some issues, please have a look at [contribution.md](https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/contributions.md).



##  Recommended Sites and Packages

* **Adversarial Attack Packages:**
  
    * [https://github.com/IBM/adversarial-robustness-toolbox](https://github.com/IBM/adversarial-robustness-toolbox): Adversarial attack and defense package made by IBM. **TensorFlow, Keras, Pyotrch available.**
    * [https://github.com/bethgelab/foolbox](https://github.com/bethgelab/foolbox): Adversarial attack package made by [Bethge Lab](http://bethgelab.org/). **TensorFlow, Pyotrch available.**
    * [https://github.com/tensorflow/cleverhans](https://github.com/tensorflow/cleverhans): Adversarial attack package made by Google Brain. **TensorFlow available.**
    * [https://github.com/BorealisAI/advertorch](https://github.com/BorealisAI/advertorch): Adversarial attack package made by [BorealisAI](https://www.borealisai.com/en/). **Pytorch available.**
    * [https://github.com/DSE-MSU/DeepRobust](https://github.com/DSE-MSU/DeepRobust): Adversarial attack (especially on GNN) package made by [BorealisAI](https://www.borealisai.com/en/). **Pytorch available.**
    * https://github.com/fra31/auto-attack: Set of attacks that is believed to be the strongest in existence. **TensorFlow, Pyotrch available.**
    
    
    
* **Adversarial Defense Leaderboard:**
  
    * [https://github.com/MadryLab/mnist_challenge](https://github.com/MadryLab/mnist_challenge)
    * [https://github.com/MadryLab/cifar10_challenge](https://github.com/MadryLab/cifar10_challenge)
    * [https://www.robust-ml.org/](https://www.robust-ml.org/)
    * [https://robust.vision/benchmark/leaderboard/](https://robust.vision/benchmark/leaderboard/)
    * https://github.com/RobustBench/robustbench
    * https://github.com/Harry24k/adversarial-defenses-pytorch
    
    
    
* **Adversarial Attack and Defense Papers:**
  
    * https://nicholas.carlini.com/writing/2019/all-adversarial-example-papers.html: A Complete List of All (arXiv) Adversarial Example Papers made by Nicholas Carlini.
    * https://github.com/chawins/Adversarial-Examples-Reading-List: Adversarial Examples Reading List made by Chawin Sitawarin.
