import torch
import math
import torch.nn.functional as F
from torch import nn
from .wide_resnet import WideResNet

class BoostingWideResNet(WideResNet):

    def __init__(self, depth=34, widen_factor=20):
        super(BoostingWideResNet, self).__init__(depth=depth,
                                                  widen_factor=widen_factor,
                                                  sub_block1=True,
                                                  bias_last=False)
        self.register_buffer(
            'mu',
            torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer(
            'sigma',
            torch.tensor([0.2471, 0.2435, 0.2616]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = F.normalize(out, p=2, dim=1)
        for _, module in self.fc.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data = F.normalize(module.weight, p=2, dim=1)
        return self.fc(out)