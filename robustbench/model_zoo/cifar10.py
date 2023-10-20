from collections import OrderedDict

# import timm
import torch
from torch import nn

from robustbench.model_zoo.architectures.dm_wide_resnet import CIFAR10_MEAN, CIFAR10_STD, \
    DMWideResNet, Swish, DMPreActResNet
from robustbench.model_zoo.architectures.resnet import Bottleneck, BottleneckChen2020AdversarialNet, \
    PreActBlock, PreActBlockV2, PreActResNet, ResNet, ResNet18, BasicBlock
from robustbench.model_zoo.architectures.resnext import CifarResNeXt, \
    ResNeXtBottleneck
from robustbench.model_zoo.architectures.resnest import ResNest152
from robustbench.model_zoo.architectures.wide_resnet import WideResNet
from robustbench.model_zoo.architectures.robust_wide_resnet import RobustWideResNet
from robustbench.model_zoo.architectures.boosting_wide_resnet import BoostingWideResNet
from robustbench.model_zoo.enums import ThreatModel
from robustbench.model_zoo.architectures.CARD_resnet import LRR_ResNet, WidePreActResNet
from robustbench.model_zoo.architectures.paf_wide_resnet import pssilu_wrn_28_10


class Hendrycks2020AugMixResNeXtNet(CifarResNeXt):

    def __init__(self, depth=29, num_classes=10, cardinality=4, base_width=32):
        super().__init__(ResNeXtBottleneck,
                         depth=depth,
                         num_classes=num_classes,
                         cardinality=cardinality,
                         base_width=base_width)
        self.register_buffer('mu', torch.tensor([0.5] * 3).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.5] * 3).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)


class Hendrycks2020AugMixWRNNet(WideResNet):

    def __init__(self, depth=40, widen_factor=2):
        super().__init__(depth=depth,
                         widen_factor=widen_factor,
                         sub_block1=False)
        self.register_buffer('mu', torch.tensor([0.5] * 3).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.5] * 3).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)


class Hendrycks2019UsingNet(WideResNet):

    def __init__(self, depth=28, widen_factor=10):
        super(Hendrycks2019UsingNet, self).__init__(depth=depth,
                                                    widen_factor=widen_factor,
                                                    sub_block1=False)

    def forward(self, x):
        x = 2. * x - 1.
        return super(Hendrycks2019UsingNet, self).forward(x)


class Rice2020OverfittingNet(WideResNet):

    def __init__(self, depth=34, widen_factor=20):
        super(Rice2020OverfittingNet, self).__init__(depth=depth,
                                                     widen_factor=widen_factor,
                                                     sub_block1=False)
        self.register_buffer(
            'mu',
            torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer(
            'sigma',
            torch.tensor([0.2471, 0.2435, 0.2616]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Rice2020OverfittingNet, self).forward(x)


class Engstrom2019RobustnessNet(ResNet):

    def __init__(self):
        super(Engstrom2019RobustnessNet,
              self).__init__(Bottleneck, [3, 4, 6, 3])
        self.register_buffer(
            'mu',
            torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer(
            'sigma',
            torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Engstrom2019RobustnessNet, self).forward(x)


class Chen2020AdversarialNet(nn.Module):

    def __init__(self):
        super(Chen2020AdversarialNet, self).__init__()
        self.branch1 = ResNet(BottleneckChen2020AdversarialNet, [3, 4, 6, 3])
        self.branch2 = ResNet(BottleneckChen2020AdversarialNet, [3, 4, 6, 3])
        self.branch3 = ResNet(BottleneckChen2020AdversarialNet, [3, 4, 6, 3])

        self.models = nn.ModuleList([self.branch1, self.branch2, self.branch3])

        self.register_buffer(
            'mu',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'sigma',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        out = (x - self.mu) / self.sigma

        out1 = self.branch1(out)
        out2 = self.branch2(out)
        out3 = self.branch3(out)

        prob1 = torch.softmax(out1, dim=1)
        prob2 = torch.softmax(out2, dim=1)
        prob3 = torch.softmax(out3, dim=1)

        return (prob1 + prob2 + prob3) / 3


class Wong2020FastNet(PreActResNet):

    def __init__(self):
        super(Wong2020FastNet, self).__init__(PreActBlock, [2, 2, 2, 2])
        self.register_buffer(
            'mu',
            torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer(
            'sigma',
            torch.tensor([0.2471, 0.2435, 0.2616]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Wong2020FastNet, self).forward(x)


class Ding2020MMANet(WideResNet):
    """
    See the appendix of the LICENSE file specifically for this model.
    """

    def __init__(self, depth=28, widen_factor=4):
        super(Ding2020MMANet, self).__init__(depth=depth,
                                             widen_factor=widen_factor,
                                             sub_block1=False)

    def forward(self, x):
        mu = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        std_min = torch.ones_like(std) / (x.shape[1] * x.shape[2] *
                                          x.shape[3])**.5
        x = (x - mu) / torch.max(std, std_min)
        return super(Ding2020MMANet, self).forward(x)


class Augustin2020AdversarialNet(ResNet):

    def __init__(self):
        super(Augustin2020AdversarialNet,
              self).__init__(Bottleneck, [3, 4, 6, 3])
        self.register_buffer(
            'mu',
            torch.tensor(
                [0.4913997551666284, 0.48215855929893703,
                 0.4465309133731618]).view(1, 3, 1, 1))
        self.register_buffer(
            'sigma',
            torch.tensor(
                [0.24703225141799082, 0.24348516474564,
                 0.26158783926049628]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Augustin2020AdversarialNet, self).forward(x)


class Augustin2020AdversarialWideNet(WideResNet):

    def __init__(self, depth=34, widen_factor=10):
        super(Augustin2020AdversarialWideNet,
              self).__init__(depth=depth,
                             widen_factor=widen_factor,
                             sub_block1=False)
        self.register_buffer(
            'mu',
            torch.tensor(
                [0.4913997551666284, 0.48215855929893703,
                 0.4465309133731618]).view(1, 3, 1, 1))
        self.register_buffer(
            'sigma',
            torch.tensor(
                [0.24703225141799082, 0.24348516474564,
                 0.26158783926049628]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Augustin2020AdversarialWideNet, self).forward(x)


class Rice2020OverfittingNetL2(PreActResNet):

    def __init__(self):
        super(Rice2020OverfittingNetL2, self).__init__(PreActBlockV2,
                                                       [2, 2, 2, 2],
                                                       bn_before_fc=True)
        self.register_buffer(
            'mu',
            torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer(
            'sigma',
            torch.tensor([0.2471, 0.2435, 0.2616]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Rice2020OverfittingNetL2, self).forward(x)


class Rony2019DecouplingNet(WideResNet):

    def __init__(self, depth=28, widen_factor=10):
        super(Rony2019DecouplingNet, self).__init__(depth=depth,
                                                    widen_factor=widen_factor,
                                                    sub_block1=False)
        self.register_buffer(
            'mu',
            torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1))
        self.register_buffer(
            'sigma',
            torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Rony2019DecouplingNet, self).forward(x)


class Kireev2021EffectivenessNet(PreActResNet):

    def __init__(self):
        super(Kireev2021EffectivenessNet, self).__init__(PreActBlockV2,
                                                         [2, 2, 2, 2],
                                                         bn_before_fc=True)
        self.register_buffer(
            'mu',
            torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer(
            'sigma',
            torch.tensor([0.2471, 0.2435, 0.2616]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Kireev2021EffectivenessNet, self).forward(x)


class Chen2020EfficientNet(WideResNet):

    def __init__(self, depth=34, widen_factor=10):
        super().__init__(depth=depth,
                         widen_factor=widen_factor,
                         sub_block1=True)
        self.register_buffer(
            'mu',
            torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer(
            'sigma',
            torch.tensor([0.2471, 0.2435, 0.2616]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)


class Diffenderfer2021CARD(LRR_ResNet):

    def __init__(self, width=128):
        super(Diffenderfer2021CARD, self).__init__(width=width)
        self.register_buffer(
            'mu',
            torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1))
        self.register_buffer(
            'sigma',
            torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)


class Diffenderfer2021CARD_Deck(nn.Module):

    def __init__(self, width=128):
        super(Diffenderfer2021CARD_Deck, self).__init__()
        self.num_cards = 6
        self.models = nn.ModuleList()

        for i in range(self.num_cards):
            self.models.append(LRR_ResNet(width=width))

        self.register_buffer(
            'mu',
            torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1))
        self.register_buffer(
            'sigma',
            torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma

        x_cl = x.clone(
        )  # clone to make sure x is not changed by inplace methods
        out_list = []
        for i in range(self.num_cards):
            # Evaluate model i at input
            out = self.models[i](x_cl)
            # Compute softmax
            out = torch.softmax(out, dim=1)
            # Append output to list of logits
            out_list.append(out)

        return torch.mean(torch.stack(out_list), dim=0)


class Diffenderfer2021CARD_Binary(WidePreActResNet):

    def __init__(self, num_classes=10):
        super(Diffenderfer2021CARD_Binary,
              self).__init__(num_classes=num_classes)
        self.register_buffer(
            'mu',
            torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1))
        self.register_buffer(
            'sigma',
            torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)


class Diffenderfer2021CARD_Deck_Binary(torch.nn.Module):

    def __init__(self, num_classes=10):
        super(Diffenderfer2021CARD_Deck_Binary, self).__init__()
        self.num_cards = 6
        self.models = nn.ModuleList()

        for i in range(self.num_cards):
            self.models.append(WidePreActResNet(num_classes=num_classes))

        self.register_buffer(
            'mu',
            torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1))
        self.register_buffer(
            'sigma',
            torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma

        x_cl = x.clone(
        )  # clone to make sure x is not changed by inplace methods
        out_list = []
        for i in range(self.num_cards):
            # Evaluate model i at input
            out = self.models[i](x_cl)
            # Compute softmax
            out = torch.softmax(out, dim=1)
            # Append output to list of logits
            out_list.append(out)

        return torch.mean(torch.stack(out_list), dim=0)


class Modas2021PRIMEResNet18(ResNet):

    def __init__(self, num_classes=10):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

        # mu & sigma are updated from weights
        self.register_buffer('mu', torch.tensor([0.5] * 3).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.5] * 3).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)


linf = OrderedDict(
    [
        ('Andriushchenko2020Understanding', {
            'model': lambda: PreActResNet(PreActBlock, [2, 2, 2, 2]),
            'gdrive_id': '1Uyvprd98bIyxfMjLdCZwm-NEJ-6GMVis',
        }),
        ('Carmon2019Unlabeled', {
            'model':
            lambda: WideResNet(depth=28, widen_factor=10, sub_block1=True),
            'gdrive_id':
            '15tUx-gkZMYx7BfEOw1GY5OKC-jECIsPQ',
        }),
        ('Sehwag2020Hydra', {
            'model':
            lambda: WideResNet(depth=28, widen_factor=10, sub_block1=True),
            'gdrive_id':
            '1pi8GHwAVkxVH41hEnf0IAJb_7y-Q8a2Y',
        }),
        ('Wang2020Improving', {
            'model':
            lambda: WideResNet(depth=28, widen_factor=10, sub_block1=True),
            'gdrive_id':
            '1T939mU4kXYt5bbvM55aT4fLBvRhyzjiQ',
        }),
        ('Hendrycks2019Using', {
            'model': Hendrycks2019UsingNet,
            'gdrive_id': '1-DcJsYw2dNEOyF9epks2QS7r9nqBDEsw',
        }),
        ('Rice2020Overfitting', {
            'model': Rice2020OverfittingNet,
            'gdrive_id': '1vC_Twazji7lBjeMQvAD9uEQxi9Nx2oG-',
        }),
        ('Zhang2019Theoretically', {
            'model':
            lambda: WideResNet(depth=34, widen_factor=10, sub_block1=True),
            'gdrive_id':
            '1hPz9QQwyM7QSuWu-ANG_uXR-29xtL8t_',
        }),
        ('Engstrom2019Robustness', {
            'model': Engstrom2019RobustnessNet,
            'gdrive_id': '1etqmQsksNIWBvBQ4r8ZFk_3FJlLWr8Rr',
        }),
        ('Chen2020Adversarial', {
            'model':
            Chen2020AdversarialNet,
            'gdrive_id': [
                '1HrG22y_A9F0hKHhh2cLLvKxsQTJTLE_y',
                '1DB2ymt0rMnsMk5hTuUzoMTpMKEKWpExd',
                '1GfgzNZcC190-IrT7056IZFDB6LfMUL9m'
            ],
        }),
        ('Huang2020Self', {
            'model':
            lambda: WideResNet(depth=34, widen_factor=10, sub_block1=True),
            'gdrive_id':
            '1nInDeIyZe2G-mJFxQJ3UoclQNomWjMgm',
        }),
        ('Pang2020Boosting', {
            'model': BoostingWideResNet,
            'gdrive_id': '1iNWOj3MP7kGe8yTAS4XnDaDXDLt0mwqw',
        }),
        ('Wong2020Fast', {
            'model': Wong2020FastNet,
            'gdrive_id': '1Re--_lf3jCEw9bnQqGkjw3J7v2tSZKrv',
        }),
        ('Ding2020MMA', {
            'model': Ding2020MMANet,
            'gdrive_id': '19Q_rIIHXsYzxZ0WcZdqT-N2OD7MfgoZ0',
        }),
        ('Zhang2019You', {
            'model':
            lambda: WideResNet(depth=34, widen_factor=10, sub_block1=True),
            'gdrive_id':
            '1kB2qqPQ8qUNmK8VKuTOhT1X4GT46kAoA',
        }),
        ('Zhang2020Attacks', {
            'model':
            lambda: WideResNet(depth=34, widen_factor=10, sub_block1=True),
            'gdrive_id':
            '1lBVvLG6JLXJgQP2gbsTxNHl6s3YAopqk',
        }),
        ('Wu2020Adversarial_extra', {
            'model':
            lambda: WideResNet(depth=28, widen_factor=10, sub_block1=True),
            'gdrive_id':
            '1-WJWpAZLlmc4gJ8XXNf7IETjnSZzaCNp',
        }),
        ('Wu2020Adversarial', {
            'model': lambda: WideResNet(depth=34, widen_factor=10),
            'gdrive_id': '13LBcgNvhFppCFG22i1xATrahFPfMgXGf',
        }),
        ('Gowal2020Uncovering_70_16', {
            'model':
            lambda: DMWideResNet(num_classes=10,
                                 depth=70,
                                 width=16,
                                 activation_fn=Swish,
                                 mean=CIFAR10_MEAN,
                                 std=CIFAR10_STD),
            'gdrive_id':
            "1DVwKclibqzniE2Ss5_g6BY77ChG8QKzl"
        }),
        ('Gowal2020Uncovering_70_16_extra', {
            'model':
            lambda: DMWideResNet(num_classes=10,
                                 depth=70,
                                 width=16,
                                 activation_fn=Swish,
                                 mean=CIFAR10_MEAN,
                                 std=CIFAR10_STD),
            'gdrive_id':
            "1GxryYj_Or-VCDca0wgiFLz4ssXSZXQoJ"
        }),
        ('Gowal2020Uncovering_34_20', {
            'model':
            lambda: DMWideResNet(num_classes=10,
                                 depth=34,
                                 width=20,
                                 activation_fn=Swish,
                                 mean=CIFAR10_MEAN,
                                 std=CIFAR10_STD),
            'gdrive_id':
            "1YWvZO1u9_yNLFNC3JYd_TVkvrRSMER1O"
        }),
        ('Gowal2020Uncovering_28_10_extra', {
            'model':
            lambda: DMWideResNet(num_classes=10,
                                 depth=28,
                                 width=10,
                                 activation_fn=Swish,
                                 mean=CIFAR10_MEAN,
                                 std=CIFAR10_STD),
            'gdrive_id':
            "1MBAWGxiZxKt-GfqEqtLcXcd3tAxPhvV2"
        }),
        ('Sehwag2021Proxy', {
            'model': lambda: WideResNet(34, 10, sub_block1=False),
            'gdrive_id': '1QFA5fPMj2Qw4aYNG33PkFqiv_RTDWvzm',
        }),
        ('Sehwag2021Proxy_R18', {
            'model': ResNet18,
            'gdrive_id': '1-ZgoSlD_AMhtXdnUElilxVXnzK2DcHuu',
        }),
        ('Sehwag2021Proxy_ResNest152', {
            'model': ResNest152,
            'gdrive_id': '1XSjtJZAvDlua6wTM6WRLvW_jON-DqLgT',
        }),
        ('Sitawarin2020Improving', {
            'model':
            lambda: WideResNet(depth=34, widen_factor=10, sub_block1=True),
            'gdrive_id':
            '12teknvo6dQGSWBaGnbNFwFO3-Y8j2eB6',
        }),
        ('Chen2020Efficient', {
            'model': Chen2020EfficientNet,
            'gdrive_id': '1c5EXpd3Kn_s6qQIbkLX3tTOOPC8VslHg',
        }),
        ('Cui2020Learnable_34_20', {
            'model':
            lambda: WideResNet(depth=34, widen_factor=20, sub_block1=True),
            'gdrive_id':
            '1y7BUxPhQjNlb4w4BUlDyYJIS4w4fsGiS'
        }),
        ('Cui2020Learnable_34_10', {
            'model':
            lambda: WideResNet(depth=34, widen_factor=10, sub_block1=True),
            'gdrive_id':
            '16s9pi_1QgMbFLISVvaVUiNfCzah6g2YV'
        }),
        ('Zhang2020Geometry', {
            'model':
            lambda: WideResNet(depth=28, widen_factor=10, sub_block1=True),
            'gdrive_id':
            '1UoG1JhbAps1MdMc6PEFiZ2yVXl_Ii5Jk'
        }),
        ('Rebuffi2021Fixing_28_10_cutmix_ddpm', {
            'model':
            lambda: DMWideResNet(num_classes=10,
                                 depth=28,
                                 width=10,
                                 activation_fn=Swish,
                                 mean=CIFAR10_MEAN,
                                 std=CIFAR10_STD),
            'gdrive_id':
            '1-0EChXbc6pOvx26O17av263bCeqIAz6s'
        }),
        ('Rebuffi2021Fixing_106_16_cutmix_ddpm', {
            'model':
            lambda: DMWideResNet(num_classes=10,
                                 depth=106,
                                 width=16,
                                 activation_fn=Swish,
                                 mean=CIFAR10_MEAN,
                                 std=CIFAR10_STD),
            'gdrive_id':
            '1-4qnkveIkeWoGdF72kpEFHETiY3y4_tF'
        }),
        ('Rebuffi2021Fixing_70_16_cutmix_ddpm', {
            'model':
            lambda: DMWideResNet(num_classes=10,
                                 depth=70,
                                 width=16,
                                 activation_fn=Swish,
                                 mean=CIFAR10_MEAN,
                                 std=CIFAR10_STD),
            'gdrive_id':
            '1-8CWRT-OFWyrz4T4s0I2mbFjPg8K_MUi'
        }),
        ('Rebuffi2021Fixing_70_16_cutmix_extra', {
            'model':
            lambda: DMWideResNet(num_classes=10,
                                 depth=70,
                                 width=16,
                                 activation_fn=Swish,
                                 mean=CIFAR10_MEAN,
                                 std=CIFAR10_STD),
            'gdrive_id':
            '1qKDTp6IJ1BUXZaRtbYuo_t0tuDl_4mLg'
        }),
        ('Sridhar2021Robust', {
            'model':
            lambda: WideResNet(depth=28, widen_factor=10, sub_block1=True),
            'gdrive_id':
            '1muDMpOyRlgJ7n2rhS2NpfFGp3rzjuIu0'
        }),
        ('Sridhar2021Robust_34_15', {
            'model':
            lambda: WideResNet(depth=34, widen_factor=15, sub_block1=True),
            'gdrive_id':
            '1-3ii3GX93YqIcmJ3VNsOgYA7ecdnSZ0Z',
        }),
        ('Rebuffi2021Fixing_R18_ddpm', {
            'model':
            lambda: DMPreActResNet(num_classes=10,
                                   depth=18,
                                   width=0,
                                   activation_fn=Swish,
                                   mean=CIFAR10_MEAN,
                                   std=CIFAR10_STD),
            'gdrive_id':
            '1--dxE66AsgBSUsuK2sXCTrsYUV9B5f95'
        }),
        ('Rade2021Helper_R18_extra', {
            'model':
            lambda: DMPreActResNet(num_classes=10,
                                   depth=18,
                                   width=0,
                                   activation_fn=Swish,
                                   mean=CIFAR10_MEAN,
                                   std=CIFAR10_STD),
            'gdrive_id':
            '1hdXk1rPJql2Oa84Kky64fMTQzng5UcTL'
        }),
        ('Rade2021Helper_R18_ddpm', {
            'model':
            lambda: DMPreActResNet(num_classes=10,
                                   depth=18,
                                   width=0,
                                   activation_fn=Swish,
                                   mean=CIFAR10_MEAN,
                                   std=CIFAR10_STD),
            'gdrive_id':
            '1f2yJUo-jxCQNk589frzriv6wPyrQEZdX'
        }),
        ('Rade2021Helper_extra', {
            'model':
            lambda: DMWideResNet(num_classes=10,
                                 depth=34,
                                 width=10,
                                 activation_fn=Swish,
                                 mean=CIFAR10_MEAN,
                                 std=CIFAR10_STD),
            'gdrive_id':
            '1GhAp-0C3ONRy9BxIe0J9vKc082vHvR7t'
        }),
        ('Rade2021Helper_ddpm', {
            'model':
            lambda: DMWideResNet(num_classes=10,
                                 depth=28,
                                 width=10,
                                 activation_fn=Swish,
                                 mean=CIFAR10_MEAN,
                                 std=CIFAR10_STD),
            'gdrive_id':
            '1AOF6LxnwgS5fCz_lVLYqs_wnUYuv6O7z'
        }),
        ('Huang2021Exploring', {
            'model':
            lambda: RobustWideResNet(num_classes=10,
                                     channel_configs=[16, 320, 640, 512],
                                     depth_configs=[5, 5, 5]),
            'gdrive_id':
            '1-2ram-xtoEidOh1SCYY5KTiyKieINkZe'
        }),
        ('Huang2021Exploring_ema', {
            'model':
            lambda: RobustWideResNet(num_classes=10,
                                     channel_configs=[16, 320, 640, 512],
                                     depth_configs=[5, 5, 5]),
            'gdrive_id':
            '1-GRwO5t9HxOS2y6RFK8QEsDXjdcgmVu6'
        }),
        ('Addepalli2021Towards_RN18', {
            'model': lambda: ResNet18(),
            'gdrive_id': '1-1DxecXz5U_xZ54DVdE-GVm71Tiox-Ri'
        }),
        ('Addepalli2021Towards_WRN34', {
            'model':
            lambda: WideResNet(num_classes=10, depth=34, sub_block1=True),
            'gdrive_id': '1-3vgjTNfSq7LSMKuayEQ-jLflAP196dB'
        }),
        ('Gowal2021Improving_70_16_ddpm_100m', {
            'model':
            lambda: DMWideResNet(num_classes=10,
                                 depth=70,
                                 width=16,
                                 activation_fn=Swish,
                                 mean=CIFAR10_MEAN,
                                 std=CIFAR10_STD),
            'gdrive_id':
            '10qBoDG_NomJBrRFeTsSwEcJF1GA-sjC4'
        }),
        ('Dai2021Parameterizing', {
            'model': lambda: pssilu_wrn_28_10(num_classes=10),
            'gdrive_id': '1eO-MNXQSAoCuNFjIbdCheprT4Beqo9Zv'
        }),
        ('Gowal2021Improving_28_10_ddpm_100m', {
            'model':
            lambda: DMWideResNet(num_classes=10,
                                 depth=28,
                                 width=10,
                                 activation_fn=Swish,
                                 mean=CIFAR10_MEAN,
                                 std=CIFAR10_STD),
            'gdrive_id':
            '12L8YE6VBgUDKyW6GMSNefSYj2gg4LEKx'
        }),
        ('Gowal2021Improving_R18_ddpm_100m', {
            'model':
            lambda: DMPreActResNet(num_classes=10,
                                   depth=18,
                                   width=0,
                                   activation_fn=Swish,
                                   mean=CIFAR10_MEAN,
                                   std=CIFAR10_STD),
            'gdrive_id':
            '1-0EuCJashqSXEkkd1DOzFA4tH8KL2kim'
        }),
        ('Chen2021LTD_WRN34_10', {
            'model':
            lambda: WideResNet(depth=34, widen_factor=10, sub_block1=False),
            'gdrive_id':
            '1-0RoQKYvHLNh7hZ71wJjSit1XtrJQo9D'
        }),
        ('Chen2021LTD_WRN34_20', {
            'model':
            lambda: WideResNet(depth=34, widen_factor=20, sub_block1=False),
            'gdrive_id':
            '1-5JmY9rapuGwCUc2nPvy7Rb_sn_dHhFm'
        }),
        ('Standard', {
            'model': lambda: WideResNet(depth=28, widen_factor=10),
            'gdrive_id': '1t98aEuzeTL8P7Kpd5DIrCoCL21BNZUhC',
        }),
        # ('Kang2021Stable', {
        #     'model': rebuffi_sodef,
        #     'gdrive_id': '1-HjG9f7wJDnNRdMQSiz8dlCI3sq5mfqj',
        # }),
        ('Jia2022LAS-AT_34_10', {
            'model': lambda: WideResNet(depth=34, widen_factor=10),
            'gdrive_id': '1-3l7xKhIPyes3O4QSz0HU6L-hfOoS0xD',
        }),
        ('Jia2022LAS-AT_70_16', {
            'model': lambda: WideResNet(depth=70, widen_factor=16),
            'gdrive_id': '1-4p-Gr0hjl8wq6qvvTza4x4a5Rmu-Bfr',
        }),
        ('Pang2022Robustness_WRN28_10', {
            'model':
            lambda: DMWideResNet(num_classes=10,
                                 depth=28,
                                 width=10,
                                 activation_fn=Swish,
                                 mean=CIFAR10_MEAN,
                                 std=CIFAR10_STD),
            'gdrive_id':
            '16ChNkterCp17BXv-xxqpfedb4u2_CjjS'
        }),
        ('Pang2022Robustness_WRN70_16', {
            'model':
            lambda: DMWideResNet(num_classes=10,
                                 depth=70,
                                 width=16,
                                 activation_fn=Swish,
                                 mean=CIFAR10_MEAN,
                                 std=CIFAR10_STD),
            'gdrive_id':
            '1uQZYUuUiL9BzaQUeXLhjr_RhoyFRrHe3'
        }),
        ('Addepalli2022Efficient_RN18', {
            'model': ResNet18,
            'gdrive_id': '1m5vhdzIUUKhDbsZdOG9z76Eyp6f4xe_f',
        }),
        ('Addepalli2022Efficient_WRN_34_10', {
            'model': lambda: WideResNet(depth=34, widen_factor=10),
            'gdrive_id': '1--dVDtZhAk4D2zMtTDwIGnImuCGxTcBA',
        }),
        # ('Debenedetti2022Light_XCiT-S12', {
        #     'model': (lambda: timm.create_model(
        #         'debenedetti2020light_xcit_s_cifar10_linf', pretrained=True)),
        #     'gdrive_id':
        #     None
        # }),
        # ('Debenedetti2022Light_XCiT-M12', {
        #     'model': (lambda: timm.create_model(
        #         'debenedetti2020light_xcit_m_cifar10_linf', pretrained=True)),
        #     'gdrive_id':
        #     None
        # }),
        # ('Debenedetti2022Light_XCiT-L12', {
        #     'model': (lambda: timm.create_model(
        #         'debenedetti2020light_xcit_l_cifar10_linf', pretrained=True)),
        #     'gdrive_id':
        #     None
        # }),
    ])

l2 = OrderedDict([
    ('Augustin2020Adversarial', {
        'model': Augustin2020AdversarialNet,
        'gdrive_id': '1oDghrzNfkStC2wr5Fq8T896yNV4wVG4d',
    }),
    ('Engstrom2019Robustness', {
        'model': Engstrom2019RobustnessNet,
        'gdrive_id': '1O8rGa6xOUIRwQ-M4ESrCjzknby8TM2ZE',
    }),
    ('Rice2020Overfitting', {
        'model': Rice2020OverfittingNetL2,
        'gdrive_id': '1jo-31utiYNBVzLM0NxUEWz0teo3Z0xa7',
    }),
    ('Rony2019Decoupling', {
        'model': Rony2019DecouplingNet,
        'gdrive_id': '1Oua2ZYSxNvoDrtlY9vTtRzyBWHziE4Uy',
    }),
    ('Standard', {
        'model': lambda: WideResNet(depth=28, widen_factor=10),
        'gdrive_id': '1t98aEuzeTL8P7Kpd5DIrCoCL21BNZUhC',
    }),
    ('Ding2020MMA', {
        'model': Ding2020MMANet,
        'gdrive_id': '13wgY0Q_eor52ltZ0PkfJx5BCZ8cLM52E',
    }),
    ('Wu2020Adversarial', {
        'model': lambda: WideResNet(depth=34, widen_factor=10),
        'gdrive_id': '1M5AZ0EZQt7d2AlTmsnqZcfx91-x7YEAV',
    }),
    ('Gowal2020Uncovering', {
        'model':
        lambda: DMWideResNet(num_classes=10,
                             depth=70,
                             width=16,
                             activation_fn=Swish,
                             mean=CIFAR10_MEAN,
                             std=CIFAR10_STD),
        'gdrive_id':
        "1QL4SNvYydjIg1uI3VP9SyNt-2kTXRisG"
    }),
    ('Gowal2020Uncovering_extra', {
        'model':
        lambda: DMWideResNet(num_classes=10,
                             depth=70,
                             width=16,
                             activation_fn=Swish,
                             mean=CIFAR10_MEAN,
                             std=CIFAR10_STD),
        'gdrive_id':
        "1pkZDCpCBShpAnx92n8PUeNOY1fSiTi0s"
    }),
    ('Sehwag2021Proxy', {
        'model': lambda: WideResNet(34, 10, sub_block1=False),
        'gdrive_id': '1UviikNzpltVFsgMuqQ8YhpmvGczGRS4S',
    }),
    ('Sehwag2021Proxy_R18', {
        'model': ResNet18,
        'gdrive_id': '1zPjjZj9wujBNkAmHHHIikem6_aIjMhXG',
    }),
    ('Rebuffi2021Fixing_70_16_cutmix_ddpm', {
        'model':
        lambda: DMWideResNet(num_classes=10,
                             depth=70,
                             width=16,
                             activation_fn=Swish,
                             mean=CIFAR10_MEAN,
                             std=CIFAR10_STD),
        'gdrive_id':
        '1-8ECIOYF4JB0ywxJOmhkefnv4TW-KuXp'
    }),
    ('Rebuffi2021Fixing_28_10_cutmix_ddpm', {
        'model':
        lambda: DMWideResNet(num_classes=10,
                             depth=28,
                             width=10,
                             activation_fn=Swish,
                             mean=CIFAR10_MEAN,
                             std=CIFAR10_STD),
        'gdrive_id':
        '1-DUKcvfDzeWwt0NK7q2XvU-dIi8up8B0'
    }),
    ('Rebuffi2021Fixing_70_16_cutmix_extra', {
        'model':
        lambda: DMWideResNet(num_classes=10,
                             depth=70,
                             width=16,
                             activation_fn=Swish,
                             mean=CIFAR10_MEAN,
                             std=CIFAR10_STD),
        'gdrive_id':
        '1JX82BDVBNO-Ffa2J37EuB8C-aFCbz708'
    }),
    ('Augustin2020Adversarial_34_10', {
        'model': Augustin2020AdversarialWideNet,
        'gdrive_id': '1qPsKS546mKcs71IEhzOS-kLpQFSFhaKL'
    }),
    ('Augustin2020Adversarial_34_10_extra', {
        'model': Augustin2020AdversarialWideNet,
        'gdrive_id': '1--1MFZja6C2iVWi9MgetYjnSIenRBLT-'
    }),
    ('Rebuffi2021Fixing_R18_cutmix_ddpm', {
        'model':
        lambda: DMPreActResNet(num_classes=10,
                               depth=18,
                               width=0,
                               activation_fn=Swish,
                               mean=CIFAR10_MEAN,
                               std=CIFAR10_STD),
        'gdrive_id':
        '1-AlwHsXU28tCOJsf9RKAZxVzbinzzQU3'
    }),
    ('Rade2021Helper_R18_ddpm', {
        'model':
        lambda: DMPreActResNet(num_classes=10,
                               depth=18,
                               width=0,
                               activation_fn=Swish,
                               mean=CIFAR10_MEAN,
                               std=CIFAR10_STD),
        'gdrive_id':
        '1VWrStAYy5CrUR18sjcpq_LKLpeqgUaoQ'
    }),
])

common_corruptions = OrderedDict([
    ('Diffenderfer2021Winning_LRR', {
        'model': Diffenderfer2021CARD,
        'gdrive_id': '1-NFL1OxfXgC0peeAV7G60ohhESO-8G0c'
    }),
    ('Diffenderfer2021Winning_LRR_CARD_Deck', {
        'model':
        Diffenderfer2021CARD_Deck,
        'gdrive_id': [
            '1-R56enDUZ3oY74zfmj8M5ogBsV2SEHFR',
            '1-_3eqvtBxTn-Afvg4fB5d4lx1cDi_AqC',
            '1-cY0IYzLQrXzTa3LQvl6d26KnocsSurs',
            '1-phHz-8gnNjN-m08pT_dagsW7Fa5-aTQ',
            '1-ryE4owBqQ1HiWS6L60eW5YTPII_88MH',
            '1-vka-h7krlJOjN7SRKMW5aDY73_WuZI2'
        ],
    }),
    ('Diffenderfer2021Winning_Binary', {
        'model': Diffenderfer2021CARD_Binary,
        'gdrive_id': '101VSovRuFA0M7idR9ioeVn-UxROdJUYu'
    }),
    ('Diffenderfer2021Winning_Binary_CARD_Deck', {
        'model':
        Diffenderfer2021CARD_Deck_Binary,
        'gdrive_id': [
            '10BOwYD-JdPguAp3dc4owWcyTI_MRWzmG',
            '10IuU66vHBNXVmzeY3VCfpRjGoMOpsGkl',
            '10WGSGQ0EHaJ0QZcoS0fAPqzlqwzGY-o2',
            '10X1mo5I6fKhlZzQQeLFbjnHvOHJupHeC',
            '10d8-swRiQutUSbSKksGwKeZfn8poYF3y',
            '10ilDm_fojXiQve_LaHApbNalBpk6v53Z'
        ],
    }),
    ('Rebuffi2021Fixing_70_16_cutmix_extra_Linf', {
        'model':
        lambda: DMWideResNet(num_classes=10,
                             depth=70,
                             width=16,
                             activation_fn=Swish,
                             mean=CIFAR10_MEAN,
                             std=CIFAR10_STD),
        'gdrive_id':
        '1qKDTp6IJ1BUXZaRtbYuo_t0tuDl_4mLg'
    }),
    ('Rebuffi2021Fixing_70_16_cutmix_extra_L2', {
        'model':
        lambda: DMWideResNet(num_classes=10,
                             depth=70,
                             width=16,
                             activation_fn=Swish,
                             mean=CIFAR10_MEAN,
                             std=CIFAR10_STD),
        'gdrive_id':
        '1JX82BDVBNO-Ffa2J37EuB8C-aFCbz708'
    }),
    ('Hendrycks2020AugMix_WRN', {
        'model': Hendrycks2020AugMixWRNNet,
        'gdrive_id': "1wy7gSRsUZzCzj8QhmTbcnwmES_2kkNph"
    }),
    ('Hendrycks2020AugMix_ResNeXt', {
        'model': Hendrycks2020AugMixResNeXtNet,
        'gdrive_id': "1uGP3nZbL3LC160kOsxwkkt6tDd4qbZT1"
    }),
    ('Kireev2021Effectiveness_Gauss50percent', {
        'model': Kireev2021EffectivenessNet,
        'gdrive_id': '1zR6lwYLkO3TFSgeqvu_CMYTq_IS-eicQ',
    }),
    ('Kireev2021Effectiveness_AugMixNoJSD', {
        'model': Kireev2021EffectivenessNet,
        'gdrive_id': '1p_1v1Oa-FSrjHTAq63QX4WtLYETkcbdH',
    }),
    ('Kireev2021Effectiveness_RLAT', {
        'model': Kireev2021EffectivenessNet,
        'gdrive_id': '16bCDA_5Rhr6qMKHRAO5W-4nu9_10kFyF',
    }),
    ('Kireev2021Effectiveness_RLATAugMixNoJSD', {
        'model': Kireev2021EffectivenessNet,
        'gdrive_id': '1hgJuvLPSVQMbUczn8qnIphONlJePsWgU',
    }),
    ('Kireev2021Effectiveness_RLATAugMix', {
        'model': Kireev2021EffectivenessNet,
        'gdrive_id': '19HNTdqJiuNyqFqIarPejniJEjZ3RQ_nj',
    }),
    ('Standard', {
        'model': lambda: WideResNet(depth=28, widen_factor=10),
        'gdrive_id': '1t98aEuzeTL8P7Kpd5DIrCoCL21BNZUhC',
    }),
    ('Addepalli2021Towards_WRN34', {
        'model': lambda: WideResNet(num_classes=10, depth=34, sub_block1=True),
        'gdrive_id': '1-3vgjTNfSq7LSMKuayEQ-jLflAP196dB'
    }),
    ('Modas2021PRIMEResNet18', {
        'model': Modas2021PRIMEResNet18,
        'gdrive_id': '13oDyqi16FeXy5j4Vm6IghnjTVqp_XF5U'
    }),
    ('Addepalli2022Efficient_WRN_34_10', {
        'model': lambda: WideResNet(depth=34, widen_factor=10),
        'gdrive_id': '1--dVDtZhAk4D2zMtTDwIGnImuCGxTcBA',
    })
])

cifar_10_models = OrderedDict([(ThreatModel.Linf, linf), (ThreatModel.L2, l2),
                               (ThreatModel.corruptions, common_corruptions)])
