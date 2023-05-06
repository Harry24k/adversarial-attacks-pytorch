from tejasiwni_models.wideresnet import WideResNet
from tejasiwni_models.convnext import ConvNeXt
from tejasiwni_models.resnext import ResNeXt
from tejasiwni_models.vit import VIT

cifar10_models = {'wide_resnet50_2':WideResNet(depth=50, num_classes=10), 
                'convnext_tiny':ConvNeXt.convnext_tiny(num_classes=10), 
                'resnext50_32x4d':ResNeXt.resnext50_32x4d(num_classes=10), 
                'vit_small_patch16_224':VIT.vit_small_patch16_224(num_classes=10)
                }

cifar100_models = {'wide_resnet50_2':WideResNet.wide_resnet50_2(num_classes=100), 
                'convnext_tiny':ConvNeXt.convnext_tiny(num_classes=100), 
                'resnext50_32x4d':ResNeXt.resnext50_32x4d(num_classes=100), 
                'vit_small_patch16_224':VIT.vit_small_patch16_224(num_classes=100)
                }