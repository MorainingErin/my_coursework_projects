# -*- coding: utf-8 -*-

# - Package imports - #
import torch.nn as nn
import torch.nn.functional as F


# - Coding Part - #
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet18"""
    def __init__(self, name, dropout=0.0):
        super(ResNet, self).__init__()
        self._name = name

        self._in_planes = 64
        self._conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self._bn1 = nn.BatchNorm2d(64)
        self._layer1 = self._make_layer(64, 2, stride=1)
        self._layer2 = self._make_layer(128, 2, stride=2)
        self._layer3 = self._make_layer(256, 2, stride=2)
        self._layer4 = self._make_layer(512, 2, stride=2)
        self._pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self._drop_out = nn.Dropout(dropout)
        self._act = nn.Linear(512, 10)

    @property
    def name(self):
        return self._name

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self._in_planes, planes, stride))
            self._in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self._bn1(self._conv1(x)))
        out = self._layer1(out)
        out = self._drop_out(out)
        out = self._layer2(out)
        out = self._drop_out(out)
        out = self._layer3(out)
        out = self._drop_out(out)
        out = self._layer4(out)
        out = self._drop_out(out)
        out = self._pool(out)
        out = out.view(out.size(0), -1)
        out = self._act(out)
        return out
