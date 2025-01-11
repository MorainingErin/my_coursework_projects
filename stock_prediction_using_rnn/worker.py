# -*- coding: utf-8 -*-

# - Package imports - #
import torch
import numpy as np
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torcheval.metrics.functional as metric
import utils
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sbn

from network import RNN, LSTM


# - Coding Part - #
class Worker:
    def __init__(self, args, train_dataset, test_dataset):
        self._mode = args.mode
        self._run_tag = args.run_tag
        self._res_dir = Path(args.res_dir)
        self._time_keeper = utils.TimeKeeper()
        self._avg_meter = ['train_loss', 'eval_loss']
        self._device = args.device

        self._epoch_start = 1
        self._epoch_end = args.epoch_num + 1
        self._iter_num = 0

        self._batch_num = args.batch_size
        self._learning_rate = args.lr
        self._learning_rate_step = args.lr_step
        self._optimizer = args.optimizer
        self._scheduler = None
        self._network = args.network
        self._loss_func = None

        self._board_writer = None
        self._log_file = None
        self._history_best = None

        self._train_dataset = train_dataset
        self._test_dataset = test_dataset
        self._dataset_type = args.dataset

    def logging(self, output_str, save=True):
        print(output_str)
        # Save to .log file
        if save and self._log_file is not None:
            log_title = f'[{self._time_keeper}]'
            with open(str(self._log_file), 'a') as file:
                file.write(log_title + output_str + '\n')
        pass
    
    def _net_load(self):
        model_name = self._res_dir / f'{self._network.name}_best.pt'
        if model_name.exists():
            state_dict = torch.load(model_name, map_location=self._device,
                                    weights_only=True)
            self._network.load_state_dict(state_dict, strict=True)
            self.logging(f'Model loaded: {model_name}')
        pass

    def _net_save(self):
        model_name = self._res_dir / f'{self._network.name}_best.pt'
        torch.save(self._network.state_dict(), model_name)
        self.logging(f'Best model updated: {model_name}')

    def _res_save(self, outputs):
        output_file_name = self._res_dir / f'{self._mode}_output.pkl'
        with open(str(output_file_name), 'wb') as fo:
            pickle.dump(outputs, fo)
        pass

    def init(self):
        # init logger
        log_path = self._res_dir / f'log-{self._mode}'
        log_path.mkdir(parents=True, exist_ok=True)
        self._board_writer = SummaryWriter(str(log_path), 'loss')
        self._log_file = self._res_dir / f'{self._mode}_{self._dataset_type}.log'
        self._log_file.unlink(missing_ok=True)
        self.logging(f'Writer path: {self._res_dir}')
        
        # init network
        feature_size = len(self._test_dataset.get_column_name())
        if self._network == 'rnn':
            self._network = RNN(self._network, input_size=feature_size, output_size=feature_size)
        elif self._network == 'lstm':
            self._network = LSTM(self._network, input_size=feature_size, output_size=feature_size)
        else:
            raise NotImplementedError(f'Unknown network {self._network}')
        num_params = sum(p.numel() for p in self._network.parameters())
        self.logging(f'Network parameters: {num_params}')

        # init losses
        self._avg_meter = {x: utils.EpochMeter(x) for x in self._avg_meter}
        if self._mode == 'train':
            self._loss_func = torch.nn.L1Loss()

        # init optimizers
        if self._mode == 'train':
            opt = torch.optim.Adam
            if self._optimizer == 'adam':
                opt = torch.optim.Adam
            elif self._optimizer == 'rmsprop':
                opt = torch.optim.RMSprop
            elif self._optimizer == 'nadam':
                opt = torch.optim.NAdam
            else:
                raise NotImplementedError(f'Unknown optimizer {self._optimizer}')
            self._optimizer = opt(self._network.parameters(), lr=self._learning_rate)

            if self._learning_rate_step == 0:
                self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self._optimizer, mode='min', factor=0.5, patience=50
                )
            else:
                gamma = 0.5 if self._learning_rate_step > 0 else 10.0
                self._learning_rate_step = abs(self._learning_rate_step)
                self._scheduler = torch.optim.lr_scheduler.StepLR(
                    self._optimizer, self._learning_rate_step, 
                    gamma=gamma, last_epoch=-1
                )
            self.logging(f'Learning rate: {self._learning_rate}, {self._learning_rate_step}')

        self.logging(f'Initializing finished. Run tag: {self._run_tag}')

    def run(self):
        if self._mode == 'train':
            for epoch_num in range(self._epoch_start, self._epoch_end):
                train_loss = self.train_epoch(epoch_num)
                val_loss = self.eval_epoch(epoch_num)
                self.report_loss(epoch_num, train_loss, val_loss)
                if self._learning_rate_step == 0:
                    self._scheduler.step(val_loss)
                else:
                    self._scheduler.step()
        elif self._mode == 'test':
            self._net_load()
            self.test_epoch()
        pass

    def report_loss(self, epoch, train_loss, val_loss):
        self._board_writer.add_scalar('epoch/lr', self._scheduler.get_last_lr()[0], epoch)
        self._board_writer.add_scalars('epoch/loss', {'train': train_loss, 'val': val_loss}, epoch)
        self._board_writer.flush()
        pass

    def train_epoch(self, epoch):
        train_loader = DataLoader(self._train_dataset, 
                                  batch_size=self._batch_num,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)
        self._network = self._network.to(self._device)
        self._network.train()

        with tqdm(total=len(train_loader)) as pbar:
            pbar.set_description(f'Epoch {epoch:02d}')
            for i, (idx, data) in enumerate(train_loader):
                input_data = data[:, :-1, :]
                gt_data = data[:, 1:, :]
                self._iter_num += 1
                self._optimizer.zero_grad()
                pred, _ = self._network(input_data)
                err = self._loss_func(pred, gt_data)
                err.backward()
                self._avg_meter['train_loss'].update(err, self._batch_num)
                self._optimizer.step()

                #
                # Reporting
                #
                pbar.update(1)
                if i % 20 == 0:
                    train_loss = self._avg_meter['train_loss'].get_iter()
                    for meter_name in self._avg_meter.keys():
                        self._avg_meter[meter_name].clear_iter()
                    self._board_writer.add_scalar('iter/loss', train_loss, self._iter_num)
                    pbar.set_postfix(loss=f'{train_loss:.3e}')
                    self._board_writer.flush()
            
        # Epoch finished
        train_loss = self._avg_meter['train_loss'].get_epoch()
        for meter_name in self._avg_meter.keys():
            self._avg_meter[meter_name].clear_epoch()
        return train_loss
    
    def eval_epoch(self, epoch):
        eval_loader = DataLoader(self._test_dataset,
                                 batch_size=self._batch_num,
                                 shuffle=False,
                                 num_workers=0,
                                 drop_last=True)
        self._network = self._network.to(self._device)
        self._network.eval()

        with torch.no_grad():
            for idx, data in eval_loader:
                input_data = data[:, :-1, :]
                gt_data = data[:, 1:, :]
                pred, _ = self._network(input_data)
                err = self._loss_func(pred, gt_data)
                self._avg_meter['eval_loss'].update(err, idx.shape[0])

        val_loss = self._avg_meter['eval_loss'].get_epoch()
        if self._history_best is None or self._history_best > val_loss:
            self._history_best = val_loss
            self._net_save()
        self.logging(f'Epoch {epoch} end. Eval_acc = {val_loss:.3e}')

        return val_loss

    def test_epoch(self):
        test_loader = DataLoader(self._test_dataset, 
                                 batch_size=1,
                                 shuffle=False,
                                 drop_last=False)
        self._network = self._network.to(self._device)
        self._network.eval()

        outputs = []

        with torch.no_grad():
            for idx, data in test_loader:
                last_hidden = None
                gts = self._test_dataset.norm2abs(idx, data[:, 1:]).squeeze(0)
                for l in range(data.shape[1] - 1):
                    pred, hidden = self._network(data[:, l:l + 1, :], last_hidden)
                    last_hidden = hidden
                    outputs.append(pred)
                outputs = torch.concat(outputs, dim=1)
                outputs = self._test_dataset.norm2abs(idx, outputs).squeeze(0)

        # Evaluate
        gts = gts.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()
        np.savetxt(self._res_dir / 'pred.csv', outputs, delimiter=',')
        
        def draw_curve(gt, pred, title):
            date = self._test_dataset.get_all_date()[1:]
            fig = plt.plot(date, gt, label='Ground Truth')
            plt.plot(date, pred, label='Prediction')
            # plt.title(title)
            plt.xlabel('Date')
            plt.ylabel(title)
            # if title == 'Volume':
            #     plt.ylim((900000.0, 4000000.0))
            # else:
            #     plt.ylim((650.0, 950.0))
            plt.xticks(np.arange(0, len(date), 5))
            plt.legend()
            self._board_writer.add_figure(f'{self._mode}/{title}', plt.gcf(), 0)
            plt.cla()
        
        average_err = np.abs(gts - outputs).mean(axis=0)
        column_name = self._test_dataset.get_column_name()
        for i, column in enumerate(column_name):
            draw_curve(gts[:, i], outputs[:, i], column)
            self.logging(f'{column} avg err: {average_err[i]:.3f}')
        self._board_writer.flush()
        return
