# -*- coding: utf-8 -*-

# - Package imports - #
import torch
import torch.nn.functional as F


# - Coding Part - #
class LSTM(torch.nn.Module):
    def __init__(self, name='LSTM', input_size=5, hidden_size=20, output_size=5):
        super(LSTM, self).__init__()
        self._name = name
        self._lstm = torch.nn.LSTM(input_size=input_size,
                                   hidden_size=hidden_size,
                                   num_layers=1,
                                   batch_first=True,
                                   dropout=0.0)
        self.out = torch.nn.Linear(hidden_size, output_size)
        self.bn = torch.nn.BatchNorm1d(hidden_size)
    
    @property
    def name(self):
        return self._name

    def forward(self, x, history=None):
        lstm_out, (h_n, c_n) = self._lstm(x, history)
        lstm_out = self.bn(lstm_out.permute(0, 2, 1))

        outs = []
        for t in range(lstm_out.shape[2]):
            outs.append(self.out(lstm_out[:, :, t]))
        out = torch.stack(outs, dim=1)

        return out, (h_n, c_n)
