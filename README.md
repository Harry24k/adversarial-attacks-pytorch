# Adversarial-Attacks-Pytorch

This is a lightweight repository of adversarial attacks for Pytorch.
There are frequently used attacks methods and some utils.
The aim is to provide use adversarial images wihout bothering.

## Usage

### Dependencies

- torch 1.2.0
- python 3.6

### Installation

- `pip install torchattacks` or
- `git clone https://github.com/Harry24k/adversairal-attacks-pytorch`

```python
import torchattacks
pgd_attack = torchattacks.PGD(model, eps = 4/255, alpha = 8/255)
adversarial_images = pgd_attack(images, labels)
```

### Precautions

* **WARNING** :: All images should be scaled to [0, 1] with transform[to.Tensor()] before used in attacks.
* **WARNING** :: All models should return ONLY ONE vector of `(N, C)` where `C = number of classes`.

### Attacks and Papers

The papers and the methods that suggested in each article with a brief summary and example.
All methods in this repository are provided as *CLASS*, but methods in each Repo are *NOT CLASS*.

* **Explaining and harnessing adversarial examples** : [Paper](https://arxiv.org/abs/1412.6572), [Repo](https://github.com/Harry24k/FGSM-pytorch)
  - FGSM

* **Adversarial Examples in the Physical World** : [Paper](https://arxiv.org/abs/1607.02533), [Repo](https://github.com/Harry24k/AEPW-pytorch)
  - IFGSM
  - IterLL

* **Ensemble Adversarial Traning : Attacks and Defences** : [Paper](https://arxiv.org/abs/1705.07204), [Repo](https://github.com/Harry24k/RFGSM-pytorch)
  - RFGSM

* **Towards Evaluating the Robustness of Neural Networks** : [Paper](https://arxiv.org/abs/1608.04644), [Repo](https://github.com/Harry24k/CW-pytorch)
  - CW(L2)

* **Towards Deep Learning Models Resistant to Adversarial Attacks** : [Paper](https://arxiv.org/abs/1706.06083), [Repo](https://github.com/Harry24k/PGD-pytorch)
  - PGD

* **Comment on "Adv-BNN: Improved Adversarial Defense through Robust Bayesian Neural Network"** : [Paper](https://arxiv.org/abs/1907.00895)
  - APGD

### Demos

* **White Box Attack with Imagenet** ([code](https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/demos/White%20Box%20Attack%20with%20Imagenet.ipynb)): 
This demo make adversarial examples with the Imagenet data to fool [Inception v3](https://arxiv.org/abs/1512.00567). However, whole Imagenet data is too large so in this demo, so it uses only '[Giant Panda](http://www.image-net.org/)'. But users are free to add other images in the Imagenet data. 

* **Black Box Attack with CIFAR10** ([code](https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/demos/Adversairal%20Training%20with%20MNIST.ipynb)): 
In this demo, there is a black box attack example with two different models. First, make adversarial datasets from a holdout model with CIFAR10. Second, use the datasets to attack a target model. An accuracy dropped from 77.77% to 5.1%. Also this code also contains 'Save & Load' example.

* **Adversairal Training with MNIST** ([code](https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/demos/Adversairal%20Training%20with%20MNIST.ipynb)): 
This demo shows how to do adversarial training with this repository. MNIST and custom model are used in this code. The adversarial training is progressed with PGD Attack, and FGSM Attack is applied to test the model. An accuracy of normal images is 96.37% and an accuracy of FGSM attack is 96.11% .


## Update Records

### ~ Version 0.3
* **New Attacks** : FGSM, IFGSM, IterLL, RFGSM, CW(LW), PGD added.
* **Demos** uploaded.

### Version 0.4
* **DO NOT USE** : 'init.py' is Omitted.

### Version 0.5
* **Package name changed** : 'attacks' changed to 'torchattacks'.
* **New Attacks** : APGD added.
* **attack.py** : 'update_model' method added.

### Version 0.6
* **Error Solved** : 
    * Before this version, even after getting an adversarial image, the model remains evaluation mode.
    * To solve this, below methods are modified.
        * '_switch_model' method is added into **attack.py**. It will automatically change model mode to the previous mode after getting adversarial images. When getting adversarial images, model is switched to evaluation mode.
        * '__call__' methods in all attack changed to forward. Instead of this, '__call__' method is added into 'attack.py'
* **attack.py** : To provide ease of changing images to uint8 from float, 'set_mode' and '_to_uint' is added.
    * 'set_mode' determines return all outputs as 'int' OR 'flaot' through '_to_uint'.
    * '_to_uint' changes all outputs to uint8.

### Version 0.7
* **.clone().detach() is used instead of .data**
* **torch.autograd.grad is used instead of .backward() and .grad** :
    * It has improved performance by 2% computation time reduction.