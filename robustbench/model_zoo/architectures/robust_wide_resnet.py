import torch
import math
import torch.nn.functional as F
from torch import nn
from .wide_resnet import WideResNet, NetworkBlock, BasicBlock


class RobustWideResNet(nn.Module):
    def __init__(self, num_classes=10, channel_configs=[16, 160, 320, 640],
                 depth_configs=[5, 5, 5], stride_config=[1, 2, 2],
                 drop_rate_config=[0.0, 0.0, 0.0]):
        super(RobustWideResNet, self).__init__()
        assert len(channel_configs) - 1 == len(depth_configs) == len(stride_config) == len(drop_rate_config)
        self.channel_configs = channel_configs
        self.depth_configs = depth_configs
        self.stride_config = stride_config

        self.stem_conv = nn.Conv2d(3, channel_configs[0], kernel_size=3,
                                   stride=1, padding=1, bias=False)
        self.blocks = nn.ModuleList([])
        for i, stride in enumerate(stride_config):
            self.blocks.append(NetworkBlock(block=BasicBlock,
                                            nb_layers=depth_configs[i],
                                            in_planes=channel_configs[i],
                                            out_planes=channel_configs[i+1],
                                            stride=stride,
                                            dropRate=drop_rate_config[i],))

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channel_configs[-1])
        self.relu = nn.ReLU(inplace=True)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channel_configs[-1], num_classes)
        self.fc_size = channel_configs[-1]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.stem_conv(x)
        for i, block in enumerate(self.blocks):
            out = block(out)
        out = self.relu(self.bn1(out))
        out = self.global_pooling(out)
        out = out.view(-1, self.fc_size)
        out = self.fc(out)
        return out
