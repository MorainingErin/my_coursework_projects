# -*- coding: utf-8 -*-

# - Package imports - #
import time
import torch
import numpy as np


# - Coding Part - #
def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class TimeKeeper:
    def __init__(self):
        self._start_time = time.time()

    def __str__(self):
        left_time = time.time() - self._start_time
        hour = int(left_time // 3600)
        left_time -= hour * 3600
        minute = int(left_time // 60)
        left_time -= minute * 60
        second = int(left_time)
        return f'{hour:02d}h:{minute:02d}m:{second:02d}s'


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self, name='DEFAULT'):
        self._name = name
        self._avg, self._count = 0.0, 0
    
    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu().mean().item()
        self._avg = (self._avg * self._count + val * n) / (self._count + n)
        self._count += n
    
    def get(self):
        return self._avg

    def clear(self):
        self._avg, self._count = 0.0, 0

    def __str__(self) -> str:
        return f'[{self.name}]{self.avg:.2f}-({self.count})'


class EpochMeter:
    """Iter average & epoch average is stored"""
    def __init__(self, name='DEFAULT'):
        self._name = name
        self._iter = AverageMeter(name)
        self._epoch = AverageMeter(name)
    
    def update(self, val, n=1):
        self._iter.update(val, n)
        self._epoch.update(val, n)
    
    def get_iter(self):
        return self._iter.get()
    
    def get_epoch(self):
        return self._epoch.get()
    
    def clear_iter(self):
        self._iter.clear()
    
    def clear_epoch(self):
        self._iter.clear()
        self._epoch.clear()
    
    def __str__(self) -> str:
        return f'local: {self._iter}; epoch: {self._epoch}'
