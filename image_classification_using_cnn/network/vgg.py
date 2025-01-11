# -*- coding: utf-8 -*-

# - Package imports - #
import torch.nn as nn
import torch.nn.functional as F


# - Coding Part - #
class VGG(nn.Module):
    """VGG 16"""
    def __init__(self, name, dropout=0.0):
        super(VGG, self).__init__()
        self._name = name

        layers = []
        in_ch = 3
        cfg =  [64, 64, 'M',            # 16 x 16
                128, 128, 'M',          # 8 x 8
                256, 256, 256, 'M',     # 4 x 4
                512, 512, 512, 'M',     # 2 x 2
                512, 512, 512, 'M']     # 1 x 1
        self._drop_out = nn.Dropout(dropout)
        for ch in cfg:
            if ch == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers += [nn.Conv2d(in_ch, ch, kernel_size=3, padding=1),
                           nn.BatchNorm2d(ch),
                           nn.ReLU(inplace=True),
                           self._drop_out]
                in_ch = ch
        layers.append(nn.AvgPool2d(kernel_size=1, stride=1))
        self._features = nn.Sequential(*layers)
        self._act = nn.Linear(512, 10)

    @property
    def name(self):
        return self._name
    
    def forward(self, x):
        x = self._features(x)
        x = self._act(x.view(x.shape[0], -1))
        return x
