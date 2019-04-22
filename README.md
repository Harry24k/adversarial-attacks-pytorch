# Adversarial-Attacks-Pytorch

This is a lightweight repository of adversarial attacks for Pytorch.
There are frequently used attacks methods and some utils.
The aim is to provide use adversarial images wihout bothering.

## Usage

### Dependencies

- torch 1.0.0
- python 3.6

### Installation

- `pip install torchattacks` or
- `git clone https://github.com/HarryK24/adversairal-attacks-pytorch`

```python
import attacks
pgd_attack = attacks.PGD(model, eps = 4/255, alpha = 8/255)
adversarial_images = pgd_attack(images, labels)
```

### Attacks and Papers

The papers and the methods that suggested in each article with a brief summary and example.
All methods in this repository are provided as *CLASS*, but methods in each Repo are *NOT CLASS*.

* **Explaining and harnessing adversarial examples** : [Paper](https://arxiv.org/abs/1412.6572), [Repo](https://github.com/HarryK24/FGSM-pytorch)
  - FGSM

* **Adversarial Examples in the Physical World** : [Paper](https://arxiv.org/abs/1607.02533), [Repo](https://github.com/HarryK24/AEPW-pytorch)
  - IFGSM
  - IterLL

* **Ensemble Adversarial Traning : Attacks and Defences** : [Paper](https://arxiv.org/abs/1705.07204), [Repo](https://github.com/HarryK24/RFGSM-pytorch)
  - RFGSM

* **Towards Evaluating the Robustness of Neural Networks** : [Paper](https://arxiv.org/abs/1608.04644), [Repo](https://github.com/HarryK24/CW-pytorch)
  - CW(L2)

* **Towards Deep Learning Models Resistant to Adversarial Attacks** : [Paper](https://arxiv.org/abs/1706.06083), [Repo](https://github.com/HarryK24/PGD-pytorch)
  - PGD


### Demos

* **Adversarial Attack with Imagenet** [code](/demos/Adversarial Attack with Imagenet.ipynb): 
This demo make adversarial examples with the Imagenet data to fool [Inception v3](https://arxiv.org/abs/1512.00567). However, whole Imagenet data is too large so in this demo, so it uses only '[Giant Panda](http://www.image-net.org/)'. But users are free to add other images in the Imagenet data. Also this code also contains 'Save & Load' example.

* **Adversarial Training with MNIST** [code](/demos/Adversairal Training with MNIST.ipynb): 
This demo shows how to do adversarial training with this repository. MNIST and custom model are used in this code. The adversarial training is progressed with PGD Attack, and FGSM Attack is applied to test the model.
