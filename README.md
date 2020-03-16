# Adversarial-Attacks-Pytorch

This is a lightweight repository of adversarial attacks for Pytorch.

There are popular attack methods and some utils.

## Table of Contents
1. [Usage](#Usage)
2. [Attacks and Papers](#Attacks-and-Papers)
3. [Demos](#Demos)
4. [Update Records](#Update-Records)

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

## Attacks and Papers

The papers and the methods with a brief summary and example.
All attacks in this repository are provided as *CLASS*.
If you want to get attacks built in *Function*, please refer below repositories.

* **Explaining and harnessing adversarial examples** : [Paper](https://arxiv.org/abs/1412.6572), [Repo](https://github.com/Harry24k/FGSM-pytorch)
  - FGSM

* **DeepFool: a simple and accurate method to fool deep neural networks** : [Paper](https://arxiv.org/abs/1511.04599)
  - DeepFool

* **Adversarial Examples in the Physical World** : [Paper](https://arxiv.org/abs/1607.02533), [Repo](https://github.com/Harry24k/AEPW-pytorch)
  - BIM
  - StepLL

* **Towards Evaluating the Robustness of Neural Networks** : [Paper](https://arxiv.org/abs/1608.04644), [Repo](https://github.com/Harry24k/CW-pytorch)
  - CW(L2)

* **Ensemble Adversarial Traning : Attacks and Defences** : [Paper](https://arxiv.org/abs/1705.07204), [Repo](https://github.com/Harry24k/RFGSM-pytorch)
  - RFGSM

* **Towards Deep Learning Models Resistant to Adversarial Attacks** : [Paper](https://arxiv.org/abs/1706.06083), [Repo](https://github.com/Harry24k/PGD-pytorch)
  - PGD
  - RPGD

* **Comment on "Adv-BNN: Improved Adversarial Defense through Robust Bayesian Neural Network"** : [Paper](https://arxiv.org/abs/1907.00895)
  - APGD

Attack | Clean | Adversarial
:---: | :---: | :---:
FGSM | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/clean.png" width="300" height="300"> | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/fgsm.png" width="300" height="300">
BIM | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/clean.png" width="300" height="300"> | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/bim.png" width="300" height="300">
StepLL | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/clean.png" width="300" height="300"> | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/stepll.png" width="300" height="300">
RFGSM | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/clean.png" width="300" height="300"> | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/rfgsm.png" width="300" height="300">
CW | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/clean.png" width="300" height="300"> | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/cw.png" width="300" height="300">
PGD | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/clean.png" width="300" height="300"> | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/pgd.png" width="300" height="300">
RPGD | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/clean.png" width="300" height="300"> | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/rpgd.png" width="300" height="300">
DeepFool | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/clean.png" width="300" height="300"> | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/deepfool.png" width="300" height="300">

## Demos

* **White Box Attack with Imagenet** ([code](https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/demos/White%20Box%20Attack%20with%20Imagenet.ipynb)): 
To make adversarial examples with the Imagenet dataset to fool [Inception v3](https://arxiv.org/abs/1512.00567). However, the Imagenet dataset is too large, so only '[Giant Panda](http://www.image-net.org/)' is used.

* **Black Box Attack with CIFAR10** ([code](https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/demos/Adversairal%20Training%20with%20MNIST.ipynb)): 
This demo provides an example of black box attack with two different models. First, make adversarial datasets from a holdout model with CIFAR10 and save it as torch dataset. Second, use the adversarial datasets to attack a target model.

* **Adversairal Training with MNIST** ([code](https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/demos/Adversairal%20Training%20with%20MNIST.ipynb)): 
This demo shows how to do adversarial training with this repository. The MNIST dataset and a custom model are used in this code. The adversarial training is performed with PGD, and then FGSM is applied to test the model.


## Update Records

### ~ Version 0.3
* **New Attacks** : FGSM, IFGSM, IterLL, RFGSM, CW(L2), PGD are added.
* **Demos** are uploaded.

### Version 0.4
* **DO NOT USE** : 'init.py' is omitted.

### Version 0.5
* **Package name changed** : 'attacks' is changed to 'torchattacks'.
* **New Attack** : APGD is added.
* **attack.py** : 'update_model' method is added.

### Version 0.6
* **Error Solved** : 
    * Before this version, even after getting an adversarial image, the model remains evaluation mode.
    * To solve this, below methods are modified.
        * '_switch_model' method is added into **attack.py**. It will automatically change model mode to the previous mode after getting adversarial images. When getting adversarial images, model is switched to evaluation mode.
        * '__call__' methods in all attack changed to forward. Instead of this, '__call__' method is added into 'attack.py'
* **attack.py** : To provide ease of changing images to uint8 from float, 'set_mode' and '_to_uint' is added.
    * 'set_mode' determines returning all outputs as 'int' OR 'flaot' through '_to_uint'.
    * '_to_uint' changes all outputs into uint8.

### Version 0.7
* **All attacks are modified**
    * clone().detach() is used instead of .data
    * torch.autograd.grad is used instead of .backward() and .grad :
        * It showed 2% reduction of computation time.
    
### Version 0.8
* **New Attack** : RPGD is added.
* **attack.py** : 'update_model' method is depreciated. Because torch models are passed by call-by-reference, we don't need to update models.
    * **cw.py** : In the process of cw attack, now masked_select uses a mask with dtype torch.bool instead of a mask with dtype torch.uint8.

### Version 0.9
* **New Attack** : DeepFool is added.
* **Some attacks are renamed** :
    * I-FGSM -> BIM
    * IterLL -> StepLL

### Version 1.0
* **attack.py** :
    * **load** : Load is depreciated. Instead, use TensorDataset and DataLoader.
    * **save** : The problem of calculating invalid accuracy when the mode of the attack set to 'int' is solved.

### Version 1.1
* **DeepFool** :
    * [**Error solved**](https://github.com/Harry24k/adversairal-attacks-pytorch/issues/2).