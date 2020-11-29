# Adversarial-Attacks-Pytorch

[![License](https://img.shields.io/github/license/Harry24k/adversarial-attacks-pytorch)](https://img.shields.io/github/license/Harry24k/adversarial-attacks-pytorch)
[![Pypi](https://img.shields.io/pypi/v/torchattacks.svg)](https://img.shields.io/pypi/v/torchattacks)
[![Documentation Status](https://readthedocs.org/projects/adversarial-attacks-pytorch/badge/?version=latest)](https://adversarial-attacks-pytorch.readthedocs.io/en/latest/)

This is a lightweight repository of adversarial attacks for Pytorch.

[Torchattacks](https://arxiv.org/abs/2010.01950) is a PyTorch library that contains adversarial attacks to generate adversarial examples and to verify the robustness of deep learning models.

<center>

|                         Clean Image                          |                      Adversarial Image                       |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/clean.png" width="300" height="300"> | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/pgd.png" width="300" height="300"> |

</center>

## Table of Contents
1. [Usage](#Usage)
2. [Attacks and Papers](#Attacks-and-Papers)
3. [Documentation](#Documentation)
4. [Expanding the Usage](#Expanding-the-Usage)
5. [Contribution](#Contribution)
6. [Recommended Sites and Packages](#Recommended-Sites-and-Packages)



## Usage

### :clipboard: Dependencies

- torch 1.2.0
- python 3.6



### :hammer: Installation

- `pip install torchattacks` or
- `git clone https://github.com/Harry24k/adversairal-attacks-pytorch`

```python
import torchattacks
atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=4)
adversarial_images = atk(images, labels)
```



###  :warning: Precautions

* **All images should be scaled to [0, 1] with transform[to.Tensor()] before used in attacks.** To make it easy to use adversarial attacks, a reverse-normalization is not included in the attack process. To apply an input normalization, please add a normalization layer to the model. Please refer to [the demo](https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/demos/White%20Box%20Attack%20(Imagenet).ipynb).

* **All models should return ONLY ONE vector of `(N, C)` where `C = number of classes`.** Considering most models in _torchvision.models_ return one vector of `(N,C)`, where `N` is the number of inputs and `C` is thenumber of classes, _torchattacks_ also only supports limited forms of output.  Please check the shape of the model’s output carefully.

* **`torch.backends.cudnn.deterministic = True` to get same adversarial examples with fixed random seed**. Some operations are non-deterministic with float tensors on GPU [[discuss]](https://discuss.pytorch.org/t/inconsistent-gradient-values-for-the-same-input/26179). If you want to get same results with same inputs, please run `torch.backends.cudnn.deterministic = True`[[ref]](https://stackoverflow.com/questions/56354461/reproducibility-and-performance-in-pytorch).




## Attacks and Papers

Implemented adversarial attacks in the papers.

The distance measure in parentheses.

* **Explaining and harnessing adversarial examples (Dec 2014)**: [Paper](https://arxiv.org/abs/1412.6572)
  - FGSM (Linf)
  
* **DeepFool: a simple and accurate method to fool deep neural networks (Nov 2015)**: [Paper](https://arxiv.org/abs/1511.04599)
  - DeepFool (L2)
  
* **Adversarial Examples in the Physical World (Jul 2016)**: [Paper](https://arxiv.org/abs/1607.02533)
  - BIM or iterative-FSGM (Linf)
  
* **Towards Evaluating the Robustness of Neural Networks (Aug 2016)**: [Paper](https://arxiv.org/abs/1608.04644)
  - CW (L2)
  
* **Ensemble Adversarial Traning: Attacks and Defences (May 2017)**: [Paper](https://arxiv.org/abs/1705.07204)
  - RFGSM (Linf)
  
* **Towards Deep Learning Models Resistant to Adversarial Attacks (Jun 2017)**: [Paper](https://arxiv.org/abs/1706.06083)
  - PGD (Linf)
  
* **Boosting Adversarial Attacks with Momentum (Oct 2017)**: [Paper](https://arxiv.org/abs/1710.06081)
  * MIFGSM (Linf) - :heart_eyes: Contributor [zhuangzi926](https://github.com/zhuangzi926)
  
* **Theoretically Principled Trade-off between Robustness and Accuracy (Jan 2019)**: [Paper](https://arxiv.org/abs/1901.08573)
  - TPGD (Linf)
  
* **Comment on "Adv-BNN: Improved Adversarial Defense through Robust Bayesian Neural Network" (Jul 2019)**: [Paper](https://arxiv.org/abs/1907.00895)
  - APGD or [EOT](https://arxiv.org/abs/1707.07397) + PGD (Linf)
  
* **Fast is better than free: Revisiting adversarial training (Jan 2020)**: [Paper](https://arxiv.org/abs/2001.03994)
  - FFGSM (Linf)
  



## Documentation

### :book: ReadTheDocs

Here is a [documentation](https://adversarial-attacks-pytorch.readthedocs.io/en/latest/index.html) for this package.



### :bell: ​Citation

If you want to cite this package, please use the following BibTex:

```
@article{kim2020torchattacks,
  title={Torchattacks: A Pytorch Repository for Adversarial Attacks},
  author={Kim, Hoki},
  journal={arXiv preprint arXiv:2010.01950},
  year={2020}
}
```



### :rocket: Demos

- **White Box Attack with ImageNet** ([code](https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/demos/White%20Box%20Attack%20(ImageNet).ipynb), [nbviewer](https://nbviewer.jupyter.org/github/Harry24k/adversarial-attacks-pytorch/blob/master/demos/White%20Box%20Attack%20%28ImageNet%29.ipynb)):  Using _torchattacks_ to make adversarial examples with [the ImageNet dataset](http://www.image-net.org/) to fool [Inception v3](https://arxiv.org/abs/1512.00567).
- **Black Box Attack with CIFAR10** ([code](https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Black%20Box%20Attack%20(CIFAR10).ipynb), [nbviewer](https://nbviewer.jupyter.org/github/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Black%20Box%20Attack%20%28CIFAR10%29.ipynb)):  This demo provides an example of black box attack with two different models. First, make adversarial datasets from a holdout model with CIFAR10 and save it as torch dataset. Second, use the adversarial datasets to attack a target model.
- **Adversairal Training with MNIST** ([code](https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/demos/Adversairal%20Training%20(MNIST).ipynb), [nbviewer](https://nbviewer.jupyter.org/github/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Adversairal%20Training%20%28MNIST%29.ipynb)):  This code shows how to do adversarial training with this repository. The MNIST dataset and a custom model are used in this code. The adversarial training is performed with PGD, and then FGSM is applied to evaluate the model.
- **Applications of MultiAttack with CIFAR10** ([code](https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Applications%20of%20MultiAttack%20(CIFAR10).ipynb), [nbviewer](https://nbviewer.jupyter.org/github/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Applications%20of%20MultiAttack%20(CIFAR10).ipynb)):  This code shows the applications of _Multiattack_. It can be used for implementing (1) Attack with random restarts, and (2) Attack on only correct examples.



## Expanding the Usage

Torchattacks supports collaboration with other attack packages.

Through expending the usage, we can use fucntions in _torchattacks_ such as _save_, _multiattack_.



###  :milky_way: AutoAttack

* https://github.com/fra31/auto-attack
* pip install git+https://github.com/fra31/auto-attack

```python
from torchattacks.attack import Attack
import autoattack

class AutoAttack(Attack):
    def __init__(self, model, eps):
        super(AutoAttack, self).__init__("AutoAttack", model)
        self.adversary = autoattack.AutoAttack(self.model, norm='Linf', eps=eps, version='standard', verbose=False)
        self._attack_mode = 'only_default'

    def forward(self, images, labels):
        adv_images = self.adversary.run_standard_evaluation(images.cuda(), labels.cuda(), bs=images.shape[0])
        return adv_images

atk = AutoAttack(model, eps=0.3)
atk.save(data_loader=test_loader, file_name="_temp.pt", accuracy=True)
```



###  :milky_way: FoolBox

* https://github.com/bethgelab/foolbox
* pip install foolbox
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
        batches = [(images[:batch_size//2], labels[:batch_size//2]), (images[batch_size//2:], labels[batch_size//2:])]
        self.init_attack.feed(model=self.fmodel, inputs=batches[0][0]) # feed 1st batch of inputs
        self.init_attack.feed(model=self.fmodel, inputs=batches[1][0]) # feed 2nd batch of inputs
        criterion = fb.Misclassification(labels)
        init_advs = self.init_attack.run(self.fmodel, images, criterion)
        
        # L2BrendelBethge
        adv_images = self.adversary.run(self.fmodel, images, labels, starting_points=init_advs)
        return adv_images

atk = L2BrendelBethge(model)
atk.save(data_loader=test_loader, file_name="_temp.pt", accuracy=True)
```



###  :milky_way: Adversarial-Robustness-Toolbox (ART)

* !git clone https://github.com/IBM/adversarial-robustness-toolbox
* pip install foolbox
* e.g., SaliencyMapMethod

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
                            model=self.model,
                            clip_values=(0, 1),
                            loss=nn.CrossEntropyLoss(),
                            optimizer=optim.Adam(self.model.parameters(), lr=0.01),
                            input_shape=(1, 28, 28),
                            nb_classes=10,
        )
        self.adversary = evasion.SaliencyMapMethod(classifier=self.classifier,
                                                   theta=theta, gamma=gamma,
                                                   batch_size=batch_size)
        self.target_map_function = lambda labels: (labels+1)%10
        self._attack_mode = 'only_default'
        
    def forward(self, images, labels):
        adv_images = self.adversary.generate(images, self.target_map_function(labels))
        return torch.tensor(adv_images).to(self.device)

atk = JSMA(model)
atk.save(data_loader=test_loader, file_name="_temp.pt", accuracy=True)
```



## Contribution

Contribution is always welcome! Use [pull requests](https://github.com/Harry24k/adversarial-attacks-pytorch/pulls) :blush:



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
