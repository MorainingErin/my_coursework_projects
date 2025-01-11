# -*- coding: utf-8 -*-

# - Package imports - #
import torch
import pandas
import seaborn as sbn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import v2


# - Coding Part - #
def data_clean(file_name):
    clean_file = file_name.parent / f'{file_name.stem}_Clean{file_name.suffix}'
    if clean_file.exists():
        return
    
    # Read and visulaize
    vis_folder = file_name.parent / f'{file_name.stem}-vis'
    vis_folder.mkdir(exist_ok=True)
    data_raw = pandas.read_csv(str(file_name), thousands=',')
    data_price = data_raw[['Date', 'Open', 'High', 'Low', 'Close']]
    data_price_scaled = data_price.copy()
    for column_name in ['Close', 'High', 'Low', 'Open']:
        data_price_scaled[column_name] /= data_price_scaled['Open']
    data_volume = data_raw[['Date', 'Volume']]

    # Draw figure
    def draw_price(out_name, data_frame):
        data_long_fmt = pandas.melt(data_frame, ['Date'])
        ax = sbn.lineplot(data_long_fmt, x='Date', y='value', hue='variable')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_xticks(np.arange(0, len(data_price), 400))
        fig = ax.get_figure()
        fig.set_size_inches((8, 6))
        fig.savefig(str(vis_folder / f'{out_name}.jpg'))
        fig.clear()

    def draw_volume(out_name, data_frame):
        ax = sbn.lineplot(data_frame, x='Date', y='Volume')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_xticks(np.arange(0, len(data_frame), 400))
        fig = ax.get_figure()
        fig.set_size_inches((8, 6))
        fig.savefig(str(vis_folder / f'{out_name}.jpg'))
        fig.clear()

    draw_price('raw_price_abs', data_price)
    draw_volume('raw_volume_abs', data_volume)
    draw_price('raw_price_scaled', data_price_scaled)

    # Create clean data
    close_thred = 1.5
    data_price_scaled_clean = data_price_scaled.copy()
    data_price_scaled_clean['Close'] = data_price_scaled_clean['Close'].apply(
        lambda x: x if x < close_thred else x / 2.0
    )
    draw_price('clean_price_scaled', data_price_scaled_clean)

    data_price_clean = data_price.copy()
    data_price_clean['Close'] = data_price_scaled_clean['Close'] * data_price_clean['Open']
    draw_price('clean_price_abs', data_price_clean)

    data_clean = pandas.merge(data_price_clean, data_volume)
    data_clean.to_csv(str(clean_file), index=False, encoding='utf-8')

    return


def create_datasets(file_name, column_names, device, seq_len, normalize, eval_split=None):
    data_raw = pandas.read_csv(str(file_name), thousands=',')
    data_np = data_raw[column_names].to_numpy().astype(np.float32)
    start_date_np = data_raw['Date'].to_numpy()

    if seq_len == -1:
        seq_len = len(data_np)

    indices = list(range(len(data_np) - seq_len + 1))
    train_indices, test_indices = None, None
    if eval_split is not None:
        np.random.shuffle(indices)
        train_num = int((1 - eval_split) * len(indices))
        train_indices = sorted(indices[:train_num])
        test_indices = sorted(indices[train_num:])
    else:
        test_indices = indices
    
    # Create all possible sequence
    data_seq = []
    for i in range(len(data_np) - seq_len + 1):
        data_seq.append(data_np[i:i + seq_len])
    data_seq = np.stack(data_seq, axis=0)
    
    def indices2dataset(given_indices):
        if given_indices is None:
            return None
        return GoogleStockDataset(data_seq[given_indices],
                                  device,
                                  column_names,
                                  start_date_np[given_indices],
                                  start_date_np,
                                  normalize)

    return indices2dataset(train_indices), indices2dataset(test_indices)


class GoogleStockDataset(torch.utils.data.Dataset):
    """Load data from given dataframe"""
    def __init__(self, data, device, column_names, start_date, all_date, normalize='minmax'):
        """
            data: [N, L, 5]
            device: "cuda:0" or "cpu"
            normalize: minmax or returns
        """
        self._device = device
        self._column_names = column_names
        self._start_date = start_date
        self._all_date = all_date
        self._data = torch.from_numpy(data).clone()

        # _base for normalization. Shape: [N, 1, 5]
        self._normalize = normalize
        self._normal_para = {}

        if self._normalize == 'minmax':
            data_dim = self._data.shape[-1]
            min_val = self._data.reshape(-1, data_dim).min(dim=0)[0].view(1, 1, data_dim)
            max_val = self._data.reshape(-1, data_dim).max(dim=0)[0].view(1, 1, data_dim)
            self._normal_para['min'] = min_val
            self._normal_para['max'] = max_val
            self._data = (self._data - min_val) / (max_val - min_val)
        elif self._normalize == 'returns':
            data_log = torch.log(self._data)
            self._normal_para['base'] = data_log[:, :1, :].clone()
            self._data[:, 1:, :] = data_log[:, 1:, :] - data_log[:, :-1, :]
            self._data[:, 0, :] = 0.0
        elif self._normalize == 'winminmax':
            data_dim = self._data.shape[-1]
            min_val = self._data.min(dim=1, keepdim=True)[0]  # [N, 1, data_dim]
            max_val = self._data.max(dim=1, keepdim=True)[0]  # [N, 1, data_dim]
            self._normal_para['min'] = min_val
            self._normal_para['max'] = max_val
            self._data = (self._data - min_val) / (max_val - min_val)
        elif self._normalize == 'log':
            self._data = torch.log(self._data)
        else:
            raise NotImplementedError(f'Unknown normalize type: {self._normalize}')

        for key in self._normal_para:
            self._normal_para[key] = self._normal_para[key].to(device)
        self._data = self._data.to(device)
        # print('pass')
        pass

    def norm2abs(self, idx, norm_data):
        if self._normalize == 'minmax':
            return norm_data * (self._normal_para['max'] - self._normal_para['min']) + self._normal_para['min']
        elif self._normalize == 'returns':
            cum_data = norm_data.clone()
            if len(norm_data.shape) == 2:
                cum_data = cum_data.cumsum(dim=0)
            elif len(norm_data.shape) == 3:
                cum_data = cum_data.cumsum(dim=1)
            return torch.exp(cum_data + self._normal_para['base'][idx])
        elif self._normalize == 'winminmax':
            min_val = self._normal_para['min'][idx]
            max_val = self._normal_para['max'][idx]
            return norm_data * (max_val - min_val) + min_val
        elif self._normalize == 'log':
            return torch.exp(norm_data)
        else:
            raise NotImplementedError(f'Unknown normalize type: {self._normalize}') 

    def get_column_name(self):
        return self._column_names
    
    def get_start_date(self):
        return self._start_date
    
    def get_all_date(self):
        return self._all_date

    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        """
            idx: [N]
            data: [N, L, 5]
        """
        return idx, self._data[idx]

