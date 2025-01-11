# -*- coding: utf-8 -*-

# - Package imports - #
import torch
import pickle
from torchvision.transforms import v2


# - Coding Part - #
class CIFARDataset(torch.utils.data.Dataset):
    """Load data from file directly"""
    def __init__(self, file_names, device, data_aug=None, meta_file_name=None):
        self._device = device

        self._data = []
        self._labels = []
        for file_name in file_names:
            with open(str(file_name), 'rb') as file:
                input_raw = pickle.load(file, encoding='bytes')
                self._data.append(torch.from_numpy(input_raw[b'data']))
                self._labels.append(torch.Tensor(input_raw[b'labels']))
        self._data = torch.cat(self._data, dim=0)
        self._labels = torch.cat(self._labels, dim=0)  # [N]
        self._total_num = self._data.shape[0]

        self._data = self._data.reshape(self._total_num, 32, 32, 3).permute(0, 3, 1, 2)  # [N, 3, 32, 32]

        # Load meta
        self._label_names = [f'Class {i}' for i in range(10)]
        if meta_file_name is not None:
            with open(str(meta_file_name), 'rb') as file:
                meta = pickle.load(file, encoding='utf-8')
            self._label_names = meta['label_names']

        # Set transforms
        steps = []
        if data_aug in ['geo', 'all']:
            steps.append(v2.RandomCrop(32, padding=4))
            steps.append(v2.RandomHorizontalFlip(p=0.5))
        if data_aug in ['color', 'all']:
            steps.append(v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1))
        steps.append(v2.ToDtype(torch.float32, scale=True))
        steps.append(v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        self._transforms = v2.Compose(steps)

        # To device
        self._data = self._data.to(self._device)
        self._labels = self._labels.long()
        self._labels = self._labels.to(self._device)
        self._updated_data = None
        self.update_data()
    
    def get_label_names(self):
        return self._label_names
    
    def get_gt(self):
        return self._labels

    def update_data(self):
        self._updated_data = self._transforms(self._data)

    def __len__(self):
        return self._total_num
    
    def __getitem__(self, idx):
        return self._updated_data[idx], self._labels[idx]
