from collections import OrderedDict

# import timm
import torch
from torch import nn

from robustbench.model_zoo.architectures.dm_wide_resnet import CIFAR100_MEAN, CIFAR100_STD, \
    DMWideResNet, Swish, DMPreActResNet
from robustbench.model_zoo.architectures.resnet import PreActBlock, PreActResNet,PreActBlockV2, \
    ResNet, BasicBlock
from robustbench.model_zoo.architectures.resnext import CifarResNeXt, ResNeXtBottleneck
from robustbench.model_zoo.architectures.wide_resnet import WideResNet
from robustbench.model_zoo.enums import ThreatModel
from robustbench.model_zoo.architectures.CARD_resnet import LRR_ResNet, WidePreActResNet
# from robustbench.model_zoo.architectures import xcit


class Chen2020EfficientNet(WideResNet):

    def __init__(self, depth=34, widen_factor=10):
        super().__init__(depth=depth,
                         widen_factor=widen_factor,
                         sub_block1=True,
                         num_classes=100)
        self.register_buffer(
            'mu',
            torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1))
        self.register_buffer(
            'sigma',
            torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)


class Wu2020AdversarialNet(WideResNet):

    def __init__(self, depth=34, widen_factor=10):
        super().__init__(depth=depth,
                         widen_factor=widen_factor,
                         sub_block1=False,
                         num_classes=100)
        self.register_buffer(
            'mu',
            torch.tensor(
                [0.5070751592371323, 0.48654887331495095,
                 0.4409178433670343]).view(1, 3, 1, 1))
        self.register_buffer(
            'sigma',
            torch.tensor(
                [0.2673342858792401, 0.2564384629170883,
                 0.27615047132568404]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)


class Rice2020OverfittingNet(PreActResNet):

    def __init__(self):
        super(Rice2020OverfittingNet, self).__init__(PreActBlock, [2, 2, 2, 2],
                                                     num_classes=100,
                                                     bn_before_fc=True,
                                                     out_shortcut=True)
        self.register_buffer(
            'mu',
            torch.tensor(
                [0.5070751592371323, 0.48654887331495095,
                 0.4409178433670343]).view(1, 3, 1, 1))
        self.register_buffer(
            'sigma',
            torch.tensor(
                [0.2673342858792401, 0.2564384629170883,
                 0.27615047132568404]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Rice2020OverfittingNet, self).forward(x)


class Hendrycks2019UsingNet(WideResNet):

    def __init__(self, depth=28, widen_factor=10):
        super(Hendrycks2019UsingNet, self).__init__(depth=depth,
                                                    widen_factor=widen_factor,
                                                    num_classes=100,
                                                    sub_block1=False)

    def forward(self, x):
        x = 2. * x - 1.
        return super(Hendrycks2019UsingNet, self).forward(x)


class Hendrycks2020AugMixResNeXtNet(CifarResNeXt):

    def __init__(self, depth=29, cardinality=4, base_width=32):
        super().__init__(ResNeXtBottleneck,
                         depth=depth,
                         num_classes=100,
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
                         sub_block1=False,
                         num_classes=100)
        self.register_buffer('mu', torch.tensor([0.5] * 3).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.5] * 3).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)


class Diffenderfer2021CARD(LRR_ResNet):

    def __init__(self, width=128, num_classes=100):
        super(Diffenderfer2021CARD, self).__init__(width=width,
                                                   num_classes=num_classes)
        self.register_buffer(
            'mu',
            torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1))
        self.register_buffer(
            'sigma',
            torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)


class Diffenderfer2021CARD_Deck(torch.nn.Module):

    def __init__(self, width=128, num_classes=100):
        super(Diffenderfer2021CARD_Deck, self).__init__()
        self.num_cards = 6
        self.models = nn.ModuleList()

        for i in range(self.num_cards):
            self.models.append(LRR_ResNet(width=width,
                                          num_classes=num_classes))

        self.register_buffer(
            'mu',
            torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1))
        self.register_buffer(
            'sigma',
            torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1))

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

    def __init__(self, num_classes=100):
        super(Diffenderfer2021CARD_Binary,
              self).__init__(num_classes=num_classes)
        self.register_buffer(
            'mu',
            torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1))
        self.register_buffer(
            'sigma',
            torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)


class Diffenderfer2021CARD_Deck_Binary(torch.nn.Module):

    def __init__(self, num_classes=100):
        super(Diffenderfer2021CARD_Deck_Binary, self).__init__()
        self.num_cards = 6
        self.models = nn.ModuleList()

        for i in range(self.num_cards):
            self.models.append(WidePreActResNet(num_classes=num_classes))

        self.register_buffer(
            'mu',
            torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1))
        self.register_buffer(
            'sigma',
            torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1))

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

    def __init__(self, num_classes=100):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        # mu & sigma are updated from weights
        self.register_buffer('mu', torch.tensor([0.5] * 3).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.5] * 3).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)


linf = OrderedDict([
    ('Gowal2020Uncovering', {
        'model':
        lambda: DMWideResNet(num_classes=100,
                             depth=70,
                             width=16,
                             activation_fn=Swish,
                             mean=CIFAR100_MEAN,
                             std=CIFAR100_STD),
        'gdrive_id':
        "16I86x2Vv_HCRKROC86G4dQKgO3Po5mT3"
    }),
    ('Gowal2020Uncovering_extra', {
        'model':
        lambda: DMWideResNet(num_classes=100,
                             depth=70,
                             width=16,
                             activation_fn=Swish,
                             mean=CIFAR100_MEAN,
                             std=CIFAR100_STD),
        'gdrive_id':
        "1LQBdwO2b391mg7VKcP6I0HIOpC6O83gn"
    }),
    ('Cui2020Learnable_34_20_LBGAT6', {
        'model':
        lambda: WideResNet(
            depth=34, widen_factor=20, num_classes=100, sub_block1=True),
        'gdrive_id':
        '1rN76st8q_32j6Uo8DI5XhcC2cwVhXBwK'
    }),
    ('Cui2020Learnable_34_10_LBGAT0', {
        'model':
        lambda: WideResNet(
            depth=34, widen_factor=10, num_classes=100, sub_block1=True),
        'gdrive_id':
        '1RnWbGxN-A-ltsfOvulr68U6i2L8ohAJi'
    }),
    ('Cui2020Learnable_34_10_LBGAT6', {
        'model':
        lambda: WideResNet(
            depth=34, widen_factor=10, num_classes=100, sub_block1=True),
        'gdrive_id':
        '1TfIgvW3BAkL8jL9J7AAWFSLW3SSzJ2AE'
    }),
    ('Chen2020Efficient', {
        'model': Chen2020EfficientNet,
        'gdrive_id': '1JEh95fvsfKireoELoVCBxOi12IPGFDUT'
    }),
    ('Wu2020Adversarial', {
        'model': Wu2020AdversarialNet,
        'gdrive_id': '1yWGvHmrgjtd9vOpV5zVDqZmeGhCgVYq7'
    }),
    ('Sehwag2021Proxy', {
        'model':
        lambda: WideResNet(
            depth=34, widen_factor=10, num_classes=100, sub_block1=False),
        'gdrive_id':
        '1ejMNF2O4xkSdrjtZt2UXUeim-y9F7Req',
    }),
    ('Sitawarin2020Improving', {
        'model':
        lambda: WideResNet(
            depth=34, widen_factor=10, num_classes=100, sub_block1=True),
        'gdrive_id':
        '1hbpwans776KM1SMbOxISkDx0KR0DW8EN'
    }),
    ('Hendrycks2019Using', {
        'model': Hendrycks2019UsingNet,
        'gdrive_id': '1If3tppQsCe5dN8Vbo9ff0tjlKQTTrShd'
    }),
    ('Rice2020Overfitting', {
        'model': Rice2020OverfittingNet,
        'gdrive_id': '1XXNZn3fZBOkD1aqNL1cvcD8zZDccyAZ6'
    }),
    ('Rebuffi2021Fixing_70_16_cutmix_ddpm', {
        'model':
        lambda: DMWideResNet(num_classes=100,
                             depth=70,
                             width=16,
                             activation_fn=Swish,
                             mean=CIFAR100_MEAN,
                             std=CIFAR100_STD),
        'gdrive_id':
        '1-GkVLo9QaRjCJl-by67xda1ySVhYxsLV'
    }),
    ('Rebuffi2021Fixing_28_10_cutmix_ddpm', {
        'model':
        lambda: DMWideResNet(num_classes=100,
                             depth=28,
                             width=10,
                             activation_fn=Swish,
                             mean=CIFAR100_MEAN,
                             std=CIFAR100_STD),
        'gdrive_id':
        '1-P7cs82Tj6UVx7Coin3tVurVKYwXWA9p'
    }),
    ('Rebuffi2021Fixing_R18_ddpm', {
        'model':
        lambda: DMPreActResNet(num_classes=100,
                               depth=18,
                               width=0,
                               activation_fn=Swish,
                               mean=CIFAR100_MEAN,
                               std=CIFAR100_STD),
        'gdrive_id':
        '1-Qcph_EXw1SCYhDIl8cwqTQQy0sJKO8N'
    }),
    ('Rade2021Helper_R18_ddpm', {
        'model':
        lambda: DMPreActResNet(num_classes=100,
                               depth=18,
                               width=0,
                               activation_fn=Swish,
                               mean=CIFAR100_MEAN,
                               std=CIFAR100_STD),
        'gdrive_id':
        '1-qUvfOjq6x4I8mZynfGtzzCH_nvqS_VQ'
    }),
    ('Addepalli2021Towards_PARN18', {
        'model':
        lambda: PreActResNet(
            PreActBlockV2, [2, 2, 2, 2], num_classes=100, bn_before_fc=True),
        'gdrive_id':
        '1-FwVya1sDvdFXr0_ZBoXEJW9ukGC7hPK',
    }),
    ('Addepalli2021Towards_WRN34', {
        'model':
        lambda: WideResNet(num_classes=100, depth=34, sub_block1=True),
        'gdrive_id': '1-9GAld_105-jWBLXL73btmfOCwAqvz7Y',
    }),
    ('Chen2021LTD_WRN34_10', {
        'model':
        lambda: WideResNet(
            depth=34, widen_factor=10, num_classes=100, sub_block1=True),
        'gdrive_id':
        '1-I4NZyULdEWH46b4EaCTxuuRo4eFXsg_'
    }),
    ('Pang2022Robustness_WRN28_10', {
        'model':
        lambda: DMWideResNet(num_classes=100,
                             depth=28,
                             width=10,
                             activation_fn=Swish,
                             mean=CIFAR100_MEAN,
                             std=CIFAR100_STD),
        'gdrive_id':
        "1VDDM_j5M4b6sZpt1Nnhkr8FER3kjE33M"
    }),
    ('Pang2022Robustness_WRN70_16', {
        'model':
        lambda: DMWideResNet(num_classes=100,
                             depth=70,
                             width=16,
                             activation_fn=Swish,
                             mean=CIFAR100_MEAN,
                             std=CIFAR100_STD),
        'gdrive_id':
        "1F3kn8KIdBVls8QuTWc3BbB83htkQeVQD",
    }),
    ('Jia2022LAS-AT_34_10', {
        'model':
        lambda: WideResNet(
            depth=34, widen_factor=10, num_classes=100, sub_block1=True),
        'gdrive_id':
        '1-338K2PUf5FTwk4cbUUeTNz247GrXaMG',
    }),
    ('Jia2022LAS-AT_34_20', {
        'model':
        lambda: WideResNet(depth=34, widen_factor=20, num_classes=100),
        'gdrive_id': '1WhRq01Yl1v8O3skkrGUBuySlptidc5a6',
    }),
    ('Addepalli2022Efficient_RN18', {
        'model': lambda: ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100),
        'gdrive_id': '1-2hnxC7lZOQDqQbum4yPbtRtTND86I5N',
    }),
    ('Addepalli2022Efficient_WRN_34_10', {
        'model':
        lambda: WideResNet(depth=34, widen_factor=10, num_classes=100),
        'gdrive_id': '1-3c-iniqNfiwGoGPHC3nSostnG6J9fDt',
    }),
    # ('Debenedetti2022Light_XCiT-S12', {
    #     'model':
    #     (lambda: timm.create_model('debenedetti2020light_xcit_s_cifar100_linf',
    #                                pretrained=True)),
    #     'gdrive_id':
    #     None
    # }),
    # ('Debenedetti2022Light_XCiT-M12', {
    #     'model':
    #     (lambda: timm.create_model('debenedetti2020light_xcit_m_cifar100_linf',
    #                                pretrained=True)),
    #     'gdrive_id':
    #     None
    # }),
    # ('Debenedetti2022Light_XCiT-L12', {
    #     'model':
    #     (lambda: timm.create_model('debenedetti2020light_xcit_l_cifar100_linf',
    #                                pretrained=True)),
    #     'gdrive_id':
    #     None
    # }),
])

common_corruptions = OrderedDict([
    ('Diffenderfer2021Winning_LRR', {
        'model': Diffenderfer2021CARD,
        'gdrive_id': '1-2egZ5WrO22A2pixw_UxOpENy7zwah8j'
    }),
    ('Diffenderfer2021Winning_LRR_CARD_Deck', {
        'model':
        Diffenderfer2021CARD_Deck,
        'gdrive_id': [
            '1-9-O8k6FZO0k-WhcIZCXvMBQLutxwF0I',
            '1-H_kInicE70twnsOaK3axVtHBV7WTalI',
            '1-MQjiJy01rc0Wt-dpgEx94pBYIPeXD6F',
            '1-VpIloQl8GePLSYbUjh_Sc0ehZgfiWny',
            '1-i6HADuWHZ8s598mvUL8dIYpL1mxM94f',
            '1-jRg4TpyIYcf-9SeG8vptu4X98VK1ZwE'
        ],
    }),
    ('Diffenderfer2021Winning_Binary', {
        'model': Diffenderfer2021CARD_Binary,
        'gdrive_id': '1-vFzi6uF6hgORX6sgJt1sKDPcr3SXUxB'
    }),
    ('Diffenderfer2021Winning_Binary_CARD_Deck', {
        'model':
        Diffenderfer2021CARD_Deck_Binary,
        'gdrive_id': [
            '107TKzt9Nd1ZBx5u-Lc2lgkiqCeiUChw_',
            '10EbQ3BxVQJ0-FyDV42fZL6DEVy5wT7D_',
            '10IRU_otxEVWNRLeG2D4UI5s6O97APCYH',
            '10PyjvWTTyziwpAUxyohkJZZrVHBTwABz',
            '10Skhbub7Uu6_WqQiyzBka4T91-5pOR-K',
            '10_thReUp-ia8Gxq1xdOAFelIHyoMWdV5'
        ],
    }),
    ('Gowal2020Uncovering_Linf', {
        'model':
        lambda: DMWideResNet(num_classes=100,
                             depth=70,
                             width=16,
                             activation_fn=Swish,
                             mean=CIFAR100_MEAN,
                             std=CIFAR100_STD),
        'gdrive_id':
        "16I86x2Vv_HCRKROC86G4dQKgO3Po5mT3"
    }),
    ('Gowal2020Uncovering_extra_Linf', {
        'model':
        lambda: DMWideResNet(num_classes=100,
                             depth=70,
                             width=16,
                             activation_fn=Swish,
                             mean=CIFAR100_MEAN,
                             std=CIFAR100_STD),
        'gdrive_id':
        "1LQBdwO2b391mg7VKcP6I0HIOpC6O83gn"
    }),
    ('Hendrycks2020AugMix_WRN', {
        'model': Hendrycks2020AugMixWRNNet,
        'gdrive_id': '1XpFFdCdU9LcDtcyNfo6_BV1RZHKKkBVE'
    }),
    ('Hendrycks2020AugMix_ResNeXt', {
        'model': Hendrycks2020AugMixResNeXtNet,
        'gdrive_id': '1ocnHbvDdOBLvgNr6K7vEYL08hUdkD1Rv'
    }),
    ('Addepalli2021Towards_PARN18', {
        'model':
        lambda: PreActResNet(
            PreActBlockV2, [2, 2, 2, 2], num_classes=100, bn_before_fc=True),
        'gdrive_id':
        '1-FwVya1sDvdFXr0_ZBoXEJW9ukGC7hPK',
    }),
    ('Addepalli2021Towards_WRN34', {
        'model':
        lambda: WideResNet(num_classes=100, depth=34, sub_block1=True),
        'gdrive_id': '1-9GAld_105-jWBLXL73btmfOCwAqvz7Y'
    }),
    ('Modas2021PRIMEResNet18', {
        'model': Modas2021PRIMEResNet18,
        'gdrive_id': '1kcohb2tBuJHa5pGSi4nAkvK-hXPSI6Hr'
    }),
    ('Addepalli2022Efficient_WRN_34_10', {
        'model':
        lambda: WideResNet(depth=34, widen_factor=10, num_classes=100),
        'gdrive_id': '1-3c-iniqNfiwGoGPHC3nSostnG6J9fDt',
    }),
])

cifar_100_models = OrderedDict([(ThreatModel.Linf, linf),
                                (ThreatModel.corruptions, common_corruptions)])
