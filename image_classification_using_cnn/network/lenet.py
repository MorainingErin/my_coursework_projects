# -*- coding: utf-8 -*-

# - Package imports - #
import torch
import torch.nn.functional as F


# - Coding Part - #
class LeNet(torch.nn.Module):
    def __init__(self, name, dropout=0.0):
        super(LeNet, self).__init__()
        self._name = name
        self.drop_out = torch.nn.Dropout(dropout)
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16*5*5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.act = torch.nn.Linear(84, 10)
    
    @property
    def name(self):
        return self._name

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.drop_out(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.drop_out(x)
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.drop_out(x)
        x = F.relu(self.fc2(x))
        x = self.drop_out(x)
        x = self.act(x)
        return x
