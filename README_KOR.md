# Adversarial-Attacks-PyTorch [KOR]

<p>
  <a href="https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/github/license/Harry24k/adversarial-attacks-pytorch?&color=brightgreen" /></a>
  <a href="https://pypi.org/project/torchattacks/"><img alt="Pypi" src="https://img.shields.io/pypi/v/torchattacks.svg?&color=orange" /></a>
  <a href="https://github.com/Harry24k/adversarial-attacks-pytorch/releases"><img alt="Latest Release" src="https://img.shields.io/github/release/Harry24k/adversarial-attacks-pytorch.svg?&color=blue" /></a>
  <a href="https://adversarial-attacks-pytorch.readthedocs.io/en/latest/"><img alt="Documentation Status" src="https://readthedocs.org/projects/adversarial-attacks-pytorch/badge/?version=latest" /></a>
</p>

|                         원본 이미지                          |                      적대적 이미지                       |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/clean.png" width="300" height="300"> | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/pgd.png" width="300" height="300"> |

[Torchattacks](https://arxiv.org/abs/2010.01950) 은 파이토치(PyTorch) 기반의 딥러닝 모델에 대한 적대적 공격(Adversarial Attack)을 구현한 패키지입니다. 파이토치와 친숙한 코드를 제공하여, 파이토치 사용자들이 좀 더 쉽게 적대적 공격에 친숙해지는 것을 목표로 하고 있습니다.


## 목차

1. [들어가기 전. 딥러닝에서의 보안과 적대적 공격](#들어가기-전.-딥러닝에서의-보안과-적대적-공격)
2. [사용 방법](#사용-방법)
5. [문서 및 데모](#문서-및-데모)
6. [인용하기](#인용하기)
7. [200% 활용하기](#200%-활용하기)
8. [기여하기](#기여하기)
9. [추천하는 다른 패키지 및 사이트](#추천하는-다른-패키지-및-사이트)

  

## 들어가기 전. 딥러닝에서의 보안과 적대적 공격

*딥러닝*은 현재 가장 각광받고 있는 기술이며, 시각(Vision) 및 청각(Audio)에 이르기까지 다양한 분야에 걸쳐 개발이 진행되고 있습니다. 그 무한한 가능성에 많은 학자들에 의해 활발하게 연구되고 있으며, 자율주행 자동차부터 인공지능 스피커까지 우리 앞에 제품으로도 모습을 드러내고 있습니다.

그런데 만약 악의를 품은 누군가가 딥러닝을 공격할 수 있다면 어떨까요? 자율주행 자동차에 심어져 있는 딥러닝 모델을 공격하여 갑자기 자동차를 정지시키거나, 인공지능 스피커를 속여 주인 모르게 결제를 진행한다면 과연 우리는 안심하고 제품과 서비스를 사용할 수 있을까요?

이렇듯 딥러닝의 성능이 향상됨에 따라 *보안(Security)*에 대한 이슈도 함께 집중받고 있습니다.



### :bulb: 적대적 공격(Adversarial Attack)의 아이디어

적대적 공격은 딥러닝을 공격하는 가장 대표적인 방법입니다. 2013년도 Szegedy et al. (Intriguing properties of neural networks)에 의해 처음 발견되어, 수많은 공격 방법이 제시되었습니다. 조금 쉽게 설명하기 위해 아래 그림을 보도록 하겠습니다.

![적대적 공격의 아이디어](https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/adv_kor.png)

위 그림의 왼쪽이 바로 우리가 딥러닝 모델을 학습(Training)하는 방법입니다. 주어진 손실함수(Loss function)을 경사 하강법(Gradient Descent)를 활용하여 줄이게끔 모델의 가중치(Weight) 혹은 파라미터(Parameter)를 변경시키게됩니다.

그럼 반대로 생각해볼 수도 있습니다. 만약 손실함수를 증가시키는 방향으로 학습하면 어떻게 될까요? 물론, 우리가 원하는 방향인 학습하는 방향이 아니라, 그 반대 방향으로 움직이게 되고 이는 당연히 아무 의미가 없게 됩니다. 하지만, 공격하고자 하는 해커에게는 이러한 방법이 매우 유용하게 이용될 수 있습니다. 이것이 바로 오른쪽 그림입니다.

오른쪽 그림에서는 경사 하강법을 반대로 활용하여, 손실함수를 증가하는 방향으로 모델의 가중치가 아닌 *이미지*를 변경시킵니다. 그럼 해당 이미지는 모델의 손실함수를 키우도록 변해서, 기존 잘 작동하던 모델을 잘 예측(Prediction)하지 못하도록 *방해*할 수 있게 됩니다. 이것이 적대적 공격의 기본적인 아이디어입니다.

물론, 원래 '개'의 이미지를 잘 맞추던 모델의 예측을 '고양이' 바꾸기 위해서 원래 '개'의 이미지를 '고양이' 이미지로 바꿔야한다면 공격은 아무 의미가 없을겁니다. 하지만, 문제는 예측을 바꾸는 데에 큰 노이즈(Noise)가 필요하지 않다는 점입니다. 본 문서의 맨 위처럼 판다(Panda)에 조금만 노이즈를 줘도 예측은 크게 빗나갑니다.



### :mag: ​적대적 공격의 분류

현재까지 다양한 공격 방법이 등장했지만, 여기서는 간편함을 위해 크게 두 가지로 나누어 서술하겠습니다.

* **Evasion Attack**: 모델의 예측(Inference)을 공격하는 것.
* **Poisoning Attack**: 모델의 학습(Training)을 공격하는 것.



#### Evasion Attack.

**Evasion Attack**은 말 그대로, 이미 학습된 모델을 공격하는 것입니다. 주로 사진이나 소리에 노이즈(Noise)를 더해 잘못된 예측을 유도합니다. 이 때 적대적 공격을 위해 사용된 노이즈를 특별히 적대적 섭동(Adversarial Perturbation)이라고도 부릅니다. 그리고, 적대적 공격을 활용해 생성된 섭동이 더해진 이미지를 적대적 예제(Adversarial Example)이라고 부르게 됩니다. 문제는 위에서 언급했던 것과 같이, 섭동이 사람의 눈에 보이지 않을 정도임에도 예측이 민감하게 바뀐다는 것입니다.

Evasion Attack의 경우 크게 **White Box**와 **Black Box**로 나뉠 수 있습니다.

* **White Box Attack**: 모델 자체에 접근 가능한 경우의 공격. (기울기 정보 활용 가능)

* **Black Box Attack**: 모델에 대한 정보가 아예 없거나, 사용된 구조 혹은 결과만 알 수 있는 경우의 공격. (기울기 정보 활용 불가능)

  * **Transfer Attack**: 대리(Surrogate) 모델을 활용하여 공격하는 방법.
  
  * **Score-based, Decision-based Attack** : 모델이 출력하는 값(Output)인 예측(Prediction)이나 확률(Probabilty)를 가지고 공격하는 방법. 



**White Box Attack**은 직접적으로 *기울기 정보를 활용* 할 수 있다는 점에서 **Black Box Attack**보다 훨씬 강력한 적대적 공격을 실행할 수 있습니다. **Torchattacks**가 다루고 있는 공격들이 바로 여기에 해당되며, 따라서 공격 시에 공격 대상이 될 *모델을 필요*로 합니다. 적대적 공격의 출발선인만큼, 다른 공격 방법들의 기반이 되는 논문이 많습니다.

> [추천 논문]
>
> Intriguing properties of neural networks
>
> Explaining and Harnessing Adversarial Examples
>
> DeepFool: a simple and accurate method to fool deep neural networks
>
> Toward evaluating the robustness of neural networks
>
> Ensemble adversarial training: Attacks and defenses
>
> Towards Deep Learning Models Resistant to Adversarial Attacks
>
> Boosting Adversarial Attacks with Momentum



**Black Box Attack**은 기울기 정보를 활용하지 못하므로, 공격자에게 *더 어려운 환경* 이라고 할 수 있습니다. 이 경우에는 공격 대상 모델과 비슷한 모델인 대리 모델을 활용하는 **Transfer Attack**이나 모델의 출력을 바탕으로 공격하는 **Score-based, Decision-based Attack** 등이 사용됩니다. 다만, 모델을 직접적으로 활용할 수 없으므로, 모델에 입력 값을 전달하고 출력을 받아오는 *쿼리(Query)*를 온라인에 요청하게 됩니다. 문제는 공격을 위해서는 다수의 쿼리가 요구되기 때문에 *많은 시간을 요구*합니다. 

> [추천 논문]
>
> Practical Black-Box Attacks against Machine Learning
>
> ZOO: Zeroth Order Optimization based Black-box Attacks to Deep Neural Networks without Training Substitute Models
>
> Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models
>
> Black-box Adversarial Attacks with Limited Queries and Information



#### Poisoning Attack.

**Poisoning Attack**은 모델의 학습을 잘못된 방향으로 이끄는 공격으로, 주로 *학습 데이터(Training Data)*를 변화시켜서 이를 이루어내게 합니다.

학습 데이터를 변화시키는 방향으로는 여러 방향이 존재하는데, 모델의 전체적인 성능을 낮추도록(Performance Degradation) 하거나, 특정 이미지 혹은 라벨(Label)만 틀리게 하도록(Targeted Poisoning) 하는 방법이 있습니다.

> [추천 논문]
>
> Towards Poisoning of Deep Learning Algorithms with Back-Gradient Optimization.
>
> Poisoning Attacks with Generative Adversarial Nets
>
> Transferable Clean-Label Poisoning Attacks on Deep Neural Nets.



### :pencil: ​적대적 공격의 연구

적대적 공격은 말 그대로 딥러닝이 실생활에 적용되었을 때에 악용될 위험이 크다는 점에서도 연구되고 있지만, 최근에는 딥러닝 모델이 왜 적대적 공격이 가능한가에 대해서 고찰하는 논문이 많습니다. 초반에는 방어 기법과 공격 기법이 번갈아 등장하면서 창과 방패의 싸움이 이어졌다면, 최근에는 어째서 조그마한 노이즈에도 모델이 민감하게 반응하는지, 어떻게하면 그러한 민감한 반응을 줄이고 *안정적(Stable)인 모델*을 만들어낼 수 있는지 등 좀 더 근원적인 부분에 대한 연구도 이루어지고 있습니다.



적대적 공격의 방어 기법으로는 학습 과정에서 적대적 예제를 만들고 활용하는 **Adversarial Training**, 특정 크기의 노이즈에 대해서는 절대 혹은 높은 확률로 공격 당하지 않게 학습하는 **Certified Training / Randomized Smoothing**, 적대적 예제를 입력부터 걸러내는 **이상 탐지(Adversarial Example Detection)** 등이 있습니다.

> [추천 논문]
>
> Towards Robust Neural Networks via Random Self-ensemble
>
> Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples
>
> Understanding Measures of Uncertainty for Adversarial Example Detection
>
> Adversarially Robust Generalization Requires More Data
>
> Robustness May Be at Odds with Accuracy
>
> On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models
>
> Theoretically Principled Trade-off between Robustness and Accuracy
>
> Adversarial Examples Are Not Bugs, They Are Features
>
> Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks





## 사용 방법

### :clipboard: 개발 환경

- torch>=1.4.0
- python>=3.6



### :hammer: 설치 방법 및 사용

- `pip install torchattacks` or
- `git clone https://github.com/Harry24k/adversairal-attacks-pytorch`

```python
import torchattacks
atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=4)
adversarial_images = atk(images, labels)
```



Torchattack은 또한 아래의 기능도 제공합니다.

<details><summary>공격 라벨 정하기</summary><p>

* Random target label:
```python
# random labels as target labels.
atk.set_mode_targeted_random(n_classses)
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

* Return to default:
```python
atk.set_mode_default()
```

</p></details>

<details><summary>반환 형식 바꾸기</summary><p>

* Return adversarial images with integer value (0-255).
```python
atk.set_return_type(type='int')
```

* Return adversarial images with float value (0-1).
```python
atk.set_return_type(type='float')
```

</p></details>

<details><summary>적대적 예제 저장하기</summary><p>
```python
atk.save(data_loader, save_path=None, verbose=True)
```
</p></details>

<details><summary>Training/Eval 모드 바꾸기</summary><p>

```python
# For RNN-based models, we cannot calculate gradients with eval mode.
# Thus, it should be changed to the training mode during the attack.
atk.set_training_mode(training=False)
```

</p></details>


<details><summary>공격 조합하기</summary><p>

* Strong attacks
```python
atk1 = torchattacks.FGSM(model, eps=8/255)
atk2 = torchattacks.PGD(model, eps=8/255, alpha=2/255, iters=40, random_start=True)
atk = torchattacks.MultiAttack([atk1, atk2])
```

* Binary serach for CW
```python
atk1 = torchattacks.CW(model, c=0.1, steps=1000, lr=0.01)
atk2 = torchattacks.CW(model, c=01, steps=1000, lr=0.01)
atk = torchattacks.MultiAttack([atk1, atk2])
```

* Random restarts
```python
atk1 = torchattacks.PGD(model, eps=8/255, alpha=2/255, iters=40, random_start=True)
atk2 = torchattacks.PGD(model, eps=8/255, alpha=2/255, iters=40, random_start=True)
atk = torchattacks.MultiAttack([atk1, atk2])
```

</p></details>




더 자세한 적용 방법은 아래 데모들을 통해 익힐 수 있습니다.

- **White Box Attack with ImageNet** ([code](https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/White%20Box%20Attack%20(ImageNet).ipynb), [nbviewer](https://nbviewer.jupyter.org/github/Harry24k/adversarial-attacks-pytorch/blob/master/demos/White%20Box%20Attack%20%28ImageNet%29.ipynb)):  ResNet-18을 ImageNet 데이터와 torchattacks을 활용하여 속이는 데모입니다.
- **Transfer Attack with CIFAR10** ([code](https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Transfer%20Attack%20(CIFAR10).ipynb), [nbviewer](https://nbviewer.jupyter.org/github/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Transfer%20Attack%20%28CIFAR10%29.ipynb)):  torchattacks을 활용하여 Transfer Attack을 실행하는 방법입니다.
- **Adversairal Training with MNIST** ([code](https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/demos/Adversairal%20Training%20(MNIST).ipynb), [nbviewer](https://nbviewer.jupyter.org/github/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Adversairal%20Training%20%28MNIST%29.ipynb)): torchattacks을 활용하여 Adversarial Training을 하는 코드입니다.



###  :warning: 주의 사항

* **모든 이미지는 transform[to.Tensor()]을 활용하여 [0, 1]로 입력되어야합니다!** 본래 PyTorch에서는 transform을 통해 지원되는 normalization을 활용하고는 합니다. 하지만, 적대적 공격의 특징 상 최대 섭동(Maximum Perturbtion) 범위를 주거나 이를 비용으로 활용하기 때문에, 입력 이미지가 [0, 1]일 때 정확히 적대적 예제를 생성할 수 있습니다. 따라서, normalization을 *데이터를 불러오는 과정*이 아니라 *모델의 안*에 삽입하여야합니다. 자세한 내용은 이 [데모](https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/White%20Box%20Attack%20(ImageNet).ipynb)를 참고 부탁드립니다.

* **모든 모델은 `(N, C)` 형태의 tensor를 출력해야합니다. 여기서 `N`은 배치(Batch)의 개수, `C`는 정답 클래스(class)의 개수입니다!** 주어지는 모델은 *torchvision.models*과의 호환을 위해서 확률 벡터로 사용될 `(N,C)` 만을 출력해야합니다. 만약 그렇지 않을 경우, 모델의 출력을 조절할 수 있는 레이어(Layer)를 이 [데모](https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Model%20with%20Multiple%20Outputs.ipynb)와 같이 추가하면 됩니다.

* **만약 매번 똑같은 적대적 예제가 나오게 하려면, `torch.backends.cudnn.deterministic = True`를 사용합니다**. GPU에서 이루어지는 연산은 non-deterministic한 경우도 있기 때문에, 기울기를 활용하는 적대적 공격은 모든 환경이 똑같더라도 항상 똑같은 값을 출력하지 않습니다 [[discuss]](https://discuss.pytorch.org/t/inconsistent-gradient-values-for-the-same-input/26179). 따라서, 이를 방지하기 위해서는  GPU의 랜덤성을 고정하도록 다음 명령어를 실행해야 합니다. `torch.backends.cudnn.deterministic = True`[[ref]](https://stackoverflow.com/questions/56354461/reproducibility-and-performance-in-pytorch).



## 인용

본 패키지를 사용하신다면 아래를 인용 부탁드립니다 :)

```
@article{kim2020torchattacks,
  title={Torchattacks: A pytorch repository for adversarial attacks},
  author={Kim, Hoki},
  journal={arXiv preprint arXiv:2010.01950},
  year={2020}
}
```



## 200% 활용하기

Torchattacks은 다른 유명한 적대적 공격 패키지와도 호환할 수 있도록 구성되어 있습니다. 특히, 다른 패키지의 공격을 *torchattacks*로 이식할경우, 적대적 예제를 저장할 수 있는 *save*나 *multiattack*을 활용하여 더 강한 공격을 만들어낼 수도 있습니다.


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



## 기여하기

어떤 종류의 기여라도 항상 감사드리며, 오류가 있다면 망설임 없이 지적 부탁드립니다. :blush:

만약, 새로운 공격을 제안하고 싶다면 [CONTRIBUTING.md](CONTRIBUTING.md)을 참고해주세요!



##  추천하는 다른 패키지 및 사이트

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



* **ETC**:

  * https://github.com/Harry24k/gnn-meta-attack: Adversarial Poisoning Attack on Graph Neural Network.
  * https://github.com/ChandlerBang/awesome-graph-attack-papers: Graph Neural Network Attack papers.

  