# -*- coding: utf-8 -*-

# - Package imports - #
import torch
import torch.nn.functional as F


# - Coding Part - #
class RNN(torch.nn.Module):
    def __init__(self, name='RNN', input_size=5, hidden_size=20, output_size=5):
        super(RNN, self).__init__()
        self._name = name
        self._rnn = torch.nn.RNN(input_size=input_size,
                                 hidden_size=hidden_size,
                                 num_layers=1,
                                 batch_first=True,
                                 dropout=0.0)
        self.out = torch.nn.Linear(hidden_size, output_size)
        self.bn = torch.nn.BatchNorm1d(hidden_size)
    
    @property
    def name(self):
        return self._name

    def forward(self, x, h_0=None):
        rnn_out, h_n = self._rnn(x, h_0)
        rnn_out = self.bn(rnn_out.permute(0, 2, 1))

        outs = []
        for t in range(rnn_out.shape[2]):
            outs.append(self.out(rnn_out[:, :, t]))
        out = torch.stack(outs, dim=1)

        return out, h_n
