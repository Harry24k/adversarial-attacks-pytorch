# Adversarial-Attacks-PyTorch

<p>
  <a href="https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/github/license/Harry24k/adversarial-attacks-pytorch?&color=brightgreen" /></a>
  <a href="https://pypi.org/project/torchattacks/"><img alt="Pypi" src="https://img.shields.io/pypi/v/torchattacks.svg?&color=orange" /></a>
  <a href="https://github.com/Harry24k/adversarial-attacks-pytorch/releases"><img alt="Latest Release" src="https://img.shields.io/github/release/Harry24k/adversarial-attacks-pytorch.svg?&color=blue" /></a>
  <a href="https://arxiv.org/abs/2010.01950"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2010.01950-f9f107.svg" /></a>
  <a href="https://adversarial-attacks-pytorch.readthedocs.io/en/latest/"><img alt="Documentation Status" src="https://readthedocs.org/projects/adversarial-attacks-pytorch/badge/?version=latest" /></a>
  <a href="https://codecov.io/gh/Harry24k/adversarial-attacks-pytorch" > 
 <img src="https://codecov.io/gh/Harry24k/adversarial-attacks-pytorch/branch/master/graph/badge.svg?token=00CQ79UTC2"/> 
  <a href="https://lgtm.com/projects/g/Harry24k/adversarial-attacks-pytorch/" > 
 <img src="https://img.shields.io/lgtm/grade/python/github/Harry24k/adversarial-attacks-pytorch?label=code%20quality"/> 
  <a href="https://pepy.tech/project/torchattacks" > 
 <img src="https://img.shields.io/pypi/dm/torchattacks?color=red"/> 
 </a>
</p>

[Torchattacks](https://adversarial-attacks-pytorch.readthedocs.io/en/latest/index.html) is a PyTorch library that provides *adversarial attacks* to generate *adversarial examples*. It contains *PyTorch-like* interface and functions that make it easier for PyTorch users to implement adversarial attacks ([README [KOR]](https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/README_KOR.md)).


```python
import torchattacks
atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=4)
# If, images are normalized:
# atk.set_normalization_used(mean=[...], std=[...])
adv_images = atk(images, labels)
```



## Table of Contents

1. [Requirements and Installation](#Requirements-and-Installation)
2. [Getting Started](#Getting-Started)
3. [Performance Comparison](#Performance-Comparison)
5. [Citation](#Citation)
7. [Contribution](#Contribution)
8. [Recommended Sites and Packages](#Recommended-Sites-and-Packages)



## Requirements and Installation

### :clipboard: Requirements

- PyTorch version >=1.4.0
- Python version >=3.6



### :hammer: Installation from pip

```
pip install torchattacks
```

### :hammer: Installation from source (recommended)

```
pip install git+https://github.com/Harry24k/adversarial-attacks-pytorch.git
```

or

```
git clone https://github.com/Harry24k/adversarial-attacks-pytorch.git
cd adversarial-attacks-pytorch/
pip install -e .
```


## Getting Started

###  :warning: Precautions
* **All models should return ONLY ONE vector of `(N, C)` where `C = number of classes`.** Considering most models in _torchvision.models_ return one vector of `(N,C)`, where `N` is the number of inputs and `C` is thenumber of classes, _torchattacks_ also only supports limited forms of output.  Please check the shape of the model’s output carefully. 
* **`torch.backends.cudnn.deterministic = True` to get same adversarial examples with fixed random seed**. Some operations are non-deterministic with float tensors on GPU [[discuss]](https://discuss.pytorch.org/t/inconsistent-gradient-values-for-the-same-input/26179). If you want to get same results with same inputs, please run `torch.backends.cudnn.deterministic = True`[[ref]](https://stackoverflow.com/questions/56354461/reproducibility-and-performance-in-pytorch).



### :rocket: Demos

* **White-box Attack on CIFAR10** ([code](https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demo/White-box%20Attack%20on%20CIFAR10.ipynb), [nbviewer](https://nbviewer.jupyter.org/github/Harry24k/adversarial-attacks-pytorch/blob/master/demo/White-box%20Attack%20on%20CIFAR10.ipynb))
* **White-box Attack on ImageNet** ([code](https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demo/White-box%20Attack%20on%20ImageNet.ipynb), [nbviewer](https://nbviewer.jupyter.org/github/Harry24k/adversarial-attacks-pytorch/blob/master/demo/White-box%20Attack%20on%20ImageNet.ipynb))
* **Transfer Attack on CIFAR10** ([code](https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demo/Transfer%20Attack%20on%20CIFAR10.ipynb), [nbviewer](https://nbviewer.jupyter.org/github/Harry24k/adversarial-attacks-pytorch/blob/master/demo/Transfer%20Attack%20on%20CIFAR10.ipynb))


### Torchattacks supports following functions:

**Targeted mode**

* Random target label:
```python
# random labels as target labels.
atk.set_mode_targeted_random()
```

* Least likely label:
```python
# label with the k-th smallest probability used as target labels.
atk.set_mode_targeted_least_likely(kth_min)
```

* By custom function:
```python
# label from mapping function
atk.set_mode_targeted_by_function(target_map_function=lambda images, labels:(labels+1)%10)
```

* By label:
```python
atk.set_mode_targeted_by_label(quiet=True)
# shift all class loops one to the right, 1=>2, 2=>3, .., 9=>0
target_labels = (labels + 1) % 10
adv_images = atk(images, target_labels)
```

* Return to default:
```python
atk.set_mode_default()
```

**Save adversarial images**

```python
# Save
atk.save(data_loader, save_path="./data.pt", verbose=True)
  
# Load
adv_loader = atk.load(load_path="./data.pt")
```

**Training/Eval during attack**

```python
# For RNN-based models, we cannot calculate gradients with eval mode.
# Thus, it should be changed to the training mode during the attack.
atk.set_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
```


**Make a set of attacks**

* Strong attacks
```python
atk1 = torchattacks.FGSM(model, eps=8/255)
atk2 = torchattacks.PGD(model, eps=8/255, alpha=2/255, iters=40, random_start=True)
atk = torchattacks.MultiAttack([atk1, atk2])
```

* Binary search for CW
```python
atk1 = torchattacks.CW(model, c=0.1, steps=1000, lr=0.01)
atk2 = torchattacks.CW(model, c=1, steps=1000, lr=0.01)
atk = torchattacks.MultiAttack([atk1, atk2])
```

* Random restarts
```python
atk1 = torchattacks.PGD(model, eps=8/255, alpha=2/255, iters=40, random_start=True)
atk2 = torchattacks.PGD(model, eps=8/255, alpha=2/255, iters=40, random_start=True)
atk = torchattacks.MultiAttack([atk1, atk2])
```



### Torchattacks also supports collaboration with other attack packages.

**FoolBox**

https://github.com/bethgelab/foolbox

```python
from torchattacks.attack import Attack
import foolbox as fb

# L2BrendelBethge
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
```

**Adversarial-Robustness-Toolbox (ART)**

https://github.com/IBM/adversarial-robustness-toolbox


```python
import torch.nn as nn
import torch.optim as optim

from torchattacks.attack import Attack

import art.attacks.evasion as evasion
from art.classifiers import PyTorchClassifier

# SaliencyMapMethod (or Jacobian based saliency map attack)
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
```

### :fire: List of implemented papers

The distance measure in parentheses.

|              Name               | Paper                                                                                                                                                     | Remark                                                                                                                 |
|:-------------------------------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
|      **FGSM**<br />(Linf)       | Explaining and harnessing adversarial examples ([Goodfellow et al., 2014](https://arxiv.org/abs/1412.6572))                                               |                                                                                                                        |
|       **BIM**<br />(Linf)       | Adversarial Examples in the Physical World ([Kurakin et al., 2016](https://arxiv.org/abs/1607.02533))                                                     | Basic iterative method or Iterative-FSGM                                                                               |
|        **CW**<br />(L2)         | Towards Evaluating the Robustness of Neural Networks ([Carlini et al., 2016](https://arxiv.org/abs/1608.04644))                                           |                                                                                                                        |
|      **RFGSM**<br />(Linf)      | Ensemble Adversarial Traning: Attacks and Defences ([Tramèr et al., 2017](https://arxiv.org/abs/1705.07204))                                              | Random initialization + FGSM                                                                                           |
|       **PGD**<br />(Linf)       | Towards Deep Learning Models Resistant to Adversarial Attacks ([Mardry et al., 2017](https://arxiv.org/abs/1706.06083))                                   | Projected Gradient Method                                                                                              |
|       **PGDL2**<br />(L2)       | Towards Deep Learning Models Resistant to Adversarial Attacks ([Mardry et al., 2017](https://arxiv.org/abs/1706.06083))                                   | Projected Gradient Method                                                                                              |
|     **MIFGSM**<br />(Linf)      | Boosting Adversarial Attacks with Momentum ([Dong et al., 2017](https://arxiv.org/abs/1710.06081))                                                        | :heart_eyes: Contributor [zhuangzi926](https://github.com/zhuangzi926), [huitailangyz](https://github.com/huitailangyz) |
|      **TPGD**<br />(Linf)       | Theoretically Principled Trade-off between Robustness and Accuracy ([Zhang et al., 2019](https://arxiv.org/abs/1901.08573))                               |                                                                                                                        |
|     **EOTPGD**<br />(Linf)      | Comment on "Adv-BNN: Improved Adversarial Defense through Robust Bayesian Neural Network" ([Zimmermann, 2019](https://arxiv.org/abs/1907.00895))          | [EOT](https://arxiv.org/abs/1707.07397)+PGD                                                                            |
|    **APGD**<br />(Linf, L2)     | Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks ([Croce et al., 2020](https://arxiv.org/abs/2001.03994)) |                                                                                                                        |
|    **APGDT**<br />(Linf, L2)    | Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks ([Croce et al., 2020](https://arxiv.org/abs/2001.03994)) | Targeted APGD                                                                                                          |
|   **FAB**<br />(Linf, L2, L1)   | Minimally distorted Adversarial Examples with a Fast Adaptive Boundary Attack ([Croce et al., 2019](https://arxiv.org/abs/1907.02044))                    |                                                                                                                        |
|   **Square**<br />(Linf, L2)    | Square Attack: a query-efficient black-box adversarial attack via random search ([Andriushchenko et al., 2019](https://arxiv.org/abs/1912.00049))         |                                                                                                                        |
| **AutoAttack**<br />(Linf, L2)  | Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks ([Croce et al., 2020](https://arxiv.org/abs/2001.03994)) | APGD+APGDT+FAB+Square                                                                                                  |
|     **DeepFool**<br />(L2)      | DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks ([Moosavi-Dezfooli et al., 2016](https://arxiv.org/abs/1511.04599))                   |                                                                                                                        |
|     **OnePixel**<br />(L0)      | One pixel attack for fooling deep neural networks ([Su et al., 2019](https://arxiv.org/abs/1710.08864))                                                   |                                                                                                                        |
|    **SparseFool**<br />(L0)     | SparseFool: a few pixels make a big difference ([Modas et al., 2019](https://arxiv.org/abs/1811.02248))                                                   |                                                                                                                        |
|     **DIFGSM**<br />(Linf)      | Improving Transferability of Adversarial Examples with Input Diversity ([Xie et al., 2019](https://arxiv.org/abs/1803.06978))                             | :heart_eyes: Contributor [taobai](https://github.com/tao-bai)                                                          |
|     **TIFGSM**<br />(Linf)      | Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks ([Dong et al., 2019](https://arxiv.org/abs/1904.02884))            | :heart_eyes: Contributor [taobai](https://github.com/tao-bai)                                                          |
| **NIFGSM**<br />(Linf) | Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks ([Lin, et al., 2022](https://arxiv.org/abs/1908.06281))                 | :heart_eyes: Contributor [Zhijin-Ge](https://github.com/Zhijin-Ge)                               |
| **SINIFGSM**<br />(Linf) | Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks ([Lin, et al., 2022](https://arxiv.org/abs/1908.06281))                 | :heart_eyes: Contributor [Zhijin-Ge](https://github.com/Zhijin-Ge)                               |
| **VMIFGSM**<br />(Linf) | Enhancing the Transferability of Adversarial Attacks through Variance Tuning ([Wang, et al., 2022](https://arxiv.org/abs/2103.15571))                 | :heart_eyes: Contributor [Zhijin-Ge](https://github.com/Zhijin-Ge)                               |
| **VNIFGSM**<br />(Linf) | Enhancing the Transferability of Adversarial Attacks through Variance Tuning ([Wang, et al., 2022](https://arxiv.org/abs/2103.15571))                 | :heart_eyes: Contributor [Zhijin-Ge](https://github.com/Zhijin-Ge)                               |
|     **Jitter**<br />(Linf)      | Exploring Misclassifications of Robust Neural Networks to Enhance Adversarial Attacks ([Schwinn, Leo, et al., 2021](https://arxiv.org/abs/2105.10304))    |                                                                                                                        |
|       **Pixle**<br />(L0)       | Pixle: a fast and effective black-box attack based on rearranging pixels ([Pomponi, Jary, et al., 2022](https://arxiv.org/abs/2202.02236))                |                                                                                                                        |
| **LGV**<br />(Linf, L2, L1, L0) | LGV: Boosting Adversarial Example Transferability from Large Geometric Vicinity ([Gubri, et al., 2022](https://arxiv.org/abs/2207.13129))                 | :heart_eyes: Contributor [Martin Gubri](https://github.com/Framartin)                               |
| **SPSA**<br />(Linf) | Adversarial Risk and the Dangers of Evaluating Against Weak Attacks ([Uesato, Jonathan, et al., 2018](https://arxiv.org/abs/1802.05666))                 | :heart_eyes: Contributor [Riko Naka](https://github.com/rikonaka)                               |
| **JSMA**<br />(L0) | The Limitations of Deep Learning in Adversarial Settings ([Papernot, Nicolas, et al., 2016](https://arxiv.org/abs/1511.07528v1))                 | :heart_eyes: Contributor [Riko Naka](https://github.com/rikonaka)                               |
| **EADL1**<br />(L1) | EAD: Elastic-Net Attacks to Deep Neural Networks ([Chen, Pin-Yu, et al., 2018](https://arxiv.org/abs/1709.04114))                 | :heart_eyes: Contributor [Riko Naka](https://github.com/rikonaka)                               |
| **EADEN**<br />(L1, L2) | EAD: Elastic-Net Attacks to Deep Neural Networks ([Chen, Pin-Yu, et al., 2018](https://arxiv.org/abs/1709.04114))                 | :heart_eyes: Contributor [Riko Naka](https://github.com/rikonaka)                               |
| **PIFGSM (PIM)**<br />(Linf) | Patch-wise Attack for Fooling Deep Neural Network ([Gao, Lianli, et al., 2020](https://arxiv.org/abs/2007.06765))                 | :heart_eyes: Contributor [Riko Naka](https://github.com/rikonaka)                               |
| **PIFGSM++ (PIM++)**<br />(Linf) | Patch-wise++ Perturbation for Adversarial Targeted Attacks ([Gao, Lianli, et al., 2021](https://arxiv.org/abs/2012.15503))                 | :heart_eyes: Contributor [Riko Naka](https://github.com/rikonaka)                               |

## Performance Comparison

For a fair comparison, [Robustbench](https://github.com/RobustBench/robustbench) is used. As for the comparison packages, currently updated and the most cited methods were selected:

* **Foolbox**: [505](https://scholar.google.com/scholar?q=Foolbox%3A%20A%20Python%20toolbox%20to%20benchmark%20the%20robustness%20of%20machine%20learning%20models.%20arXiv%202018) citations and last update 2022.10.
* **ART**: [262](https://scholar.google.com/scholar?cluster=5391305326811305758&hl=ko&as_sdt=0,5&sciodt=0,5) citations and last update 2022.10.
      

Robust accuracy against each attack and elapsed time on the first 50 images of CIFAR10. For L2 attacks, the average L2 distances between adversarial images and the original images are recorded. All experiments were done on GeForce RTX 2080. For the latest version, please refer to here ([code](https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Performance%20Comparison%20(CIFAR10).ipynb), [nbviewer](https://nbviewer.jupyter.org/github/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Performance%20Comparison%20(CIFAR10).ipynb)).

|  **Attack**  |     **Package**     |     Standard |     [Wong2020Fast](https://arxiv.org/abs/2001.03994) |     [Rice2020Overfitting](https://arxiv.org/abs/2002.11569) |     **Remark**     |
| :----------------: | :-----------------: | -------------------------------------------: | -------------------------------------------: | ---------------------------------------------: | :----------------: |
|      **FGSM** (Linf)      |    Torchattacks     | 34% (54ms) |                                 **48% (5ms)** |                                    62% (82ms) |                    |
|  | **Foolbox<sup>*</sup>** | **34% (15ms)** |                                     48% (8ms) |                  **62% (30ms)** |                    |
|                    |         ART         | 34% (214ms) |                                     48% (59ms) |                                   62% (768ms) |                    |
| **PGD** (Linf) |    **Torchattacks** | **0% (174ms)** |                               **44% (52ms)** |            **58% (1348ms)** | :crown: ​**Fastest** |
|                    | Foolbox<sup>*</sup> | 0% (354ms) |                                  44% (56ms) |              58% (1856ms) |                    |
|                    |         ART         | 0% (1384 ms) |                                   44% (437ms) |                58% (4704ms) |                    |
| **CW<sup>† </sup>**(L2) |    **Torchattacks** | **0% / 0.40<br /> (2596ms)** |                **14% / 0.61 <br />(3795ms)** | **22% / 0.56<br />(43484ms)** | :crown: ​**Highest Success Rate** <br /> :crown: **Fastest** |
|                    | Foolbox<sup>*</sup> | 0% / 0.40<br /> (2668ms) |                   32% / 0.41 <br />(3928ms) |                34% / 0.43<br />(44418ms) |  |
|                    |         ART         | 0% / 0.59<br /> (196738ms) |                 24% / 0.70 <br />(66067ms) | 26% / 0.65<br />(694972ms) |  |
| **PGD** (L2) |    **Torchattacks** | **0% / 0.41 (184ms)** |                  **68% / 0.5<br /> (52ms)** |                  **70% / 0.5<br />(1377ms)** | :crown: **Fastest** |
|                    | Foolbox<sup>*</sup> | 0% / 0.41 (396ms) |                       68% / 0.5<br /> (57ms) |                     70% / 0.5<br /> (1968ms) |                    |
|                    |         ART         | 0% / 0.40 (1364ms) |                       68% / 0.5<br /> (429ms) | 70% / 0.5<br /> (4777ms) |                           |

<sup>*</sup> Note that Foolbox returns accuracy and adversarial images simultaneously, thus the *actual* time for generating adversarial images  might be shorter than the records.

<sup>**†**</sup>Considering that the binary search algorithm for const `c` can be time-consuming, torchattacks supports MutliAttack for grid searching `c`.



To push further, I introduce [**Rai-toolbox**](https://scholar.google.com/scholar_lookup?arxiv_id=2201.05647), which is newly added package!

| Attack      | Package      | Time/step (accuracy) |
| ----------- | ------------ | -------------------- |
| FGSM (Linf) | rai-toolbox  | **58 ms** (0%)       |
|             | Torchattacks | 81 ms (0%)           |
|             | Foolbox      | 105 ms (0%)          |
|             | ART          | 83 ms (0%)           |
| PGD (Linf)  | rai-toolbox  | **58 ms** (44%)      |
|             | Torchattacks | 79 ms (44%)          |
|             | Foolbox      | 82 ms (44%)          |
|             | ART          | 90 ms (44%)          |
| PGD (L2)    | rai-toolbox  | **58 ms** (70%)      |
|             | Torchattacks | 81 ms (70%)          |
|             | Foolbox      | 82 ms (70%)          |
|             | ART          | 89 ms (70%)          |

> The rai-toolbox takes a unique approach to gradient-based perturbations: they are implemented in terms of [parameter-transforming optimizers](https://mit-ll-responsible-ai.github.io/responsible-ai-toolbox/ref_optim.html) and [perturbation models](https://mit-ll-responsible-ai.github.io/responsible-ai-toolbox/ref_perturbation.html). This enables users to implement diverse algorithms (like [universal perturbations](https://mit-ll-responsible-ai.github.io/responsible-ai-toolbox/how_to/univ_adv_pert.html) and [concept probing with sparse gradients](https://mit-ll-responsible-ai.github.io/responsible-ai-toolbox/tutorials/ImageNet-Concept-Probing.html)) using the same paradigm as a standard PGD attack.



## Citation
If you use this package, please cite the following BibTex ([SemanticScholar](https://www.semanticscholar.org/paper/Torchattacks-%3A-A-Pytorch-Repository-for-Adversarial-Kim/1f4b3283faf534ef92d7d7fa798b26480605ead9), [GoogleScholar](https://scholar.google.com/scholar?cluster=10203998516567946917&hl=ko&as_sdt=2005&sciodt=0,5)):

```
@article{kim2020torchattacks,
  title={Torchattacks: A pytorch repository for adversarial attacks},
  author={Kim, Hoki},
  journal={arXiv preprint arXiv:2010.01950},
  year={2020}
}
```



##  Recommended Sites and Packages

* **Adversarial Attack Packages:**
  
    * [https://github.com/IBM/adversarial-robustness-toolbox](https://github.com/IBM/adversarial-robustness-toolbox): Adversarial attack and defense package made by IBM. **TensorFlow, Keras, Pytorch available.**
    * [https://github.com/bethgelab/foolbox](https://github.com/bethgelab/foolbox): Adversarial attack package made by [Bethge Lab](http://bethgelab.org/). **TensorFlow, Pytorch available.**
    * [https://github.com/tensorflow/cleverhans](https://github.com/tensorflow/cleverhans): Adversarial attack package made by Google Brain. **TensorFlow available.**
    * [https://github.com/BorealisAI/advertorch](https://github.com/BorealisAI/advertorch): Adversarial attack package made by [BorealisAI](https://www.borealisai.com/en/). **Pytorch available.**
    * [https://github.com/DSE-MSU/DeepRobust](https://github.com/DSE-MSU/DeepRobust): Adversarial attack (especially on GNN) package made by [BorealisAI](https://www.borealisai.com/en/). **Pytorch available.**
    * https://github.com/fra31/auto-attack: Set of attacks that is believed to be the strongest in existence. **TensorFlow, Pytorch available.**
    * https://github.com/mit-ll-responsible-ai/responsible-ai-toolbox/: PyTorch-centric tools for evaluating and enhancing both the robustness and the explainability of AI models. **Pytorch available.**
    
    
    
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
