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

from network import LeNet, ResNet, VGG


# - Coding Part - #
class Worker:
    def __init__(self, args, train_dataset, test_dataset):
        seed = args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self._mode = args.mode
        self._run_tag = args.run_tag
        self._res_dir = Path(args.res_dir)
        self._time_keeper = utils.TimeKeeper()
        self._avg_meter = ['loss', 'train_acc']
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
        self._drop_out_rate = args.dropout
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
        log_path = self._res_dir / 'log'
        log_path.mkdir(parents=True, exist_ok=True)
        self._board_writer = SummaryWriter(str(log_path), 'loss')
        self._log_file = self._res_dir / f'{self._mode}_{self._dataset_type}.log'
        self._log_file.unlink(missing_ok=True)
        self.logging(f'Writer path: {self._res_dir}')
        
        # init network
        if self._network == 'vgg':
            self._network = VGG(self._network, self._drop_out_rate)
        elif self._network == 'lenet':
            self._network = LeNet(self._network, self._drop_out_rate)
        elif self._network == 'resnet':
            self._network = ResNet(self._network, self._drop_out_rate)
        else:
            raise NotImplementedError(f'Unknown network {self._network}')
        num_params = sum(p.numel() for p in self._network.parameters())
        self.logging(f'Network parameters: {num_params}')

        # init losses
        self._avg_meter = {x: utils.EpochMeter(x) for x in self._avg_meter}
        if self._mode == 'train':
            self._loss_func = torch.nn.CrossEntropyLoss()

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

            # Scheduler:
            if self._learning_rate_step == 0:
                self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self._optimizer, mode='max', factor=0.5, patience=50
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
                total_loss, train_acc = self.train_epoch(epoch_num)
                val_acc = self.test_epoch(epoch_num, vis_flag=(epoch_num == self._epoch_end - 1))
                self.report_loss(epoch_num, total_loss, train_acc, val_acc)
                self._scheduler.step(val_acc)
        elif self._mode == 'test':
            self._net_load()
            val_acc = self.test_epoch(self._epoch_end, vis_flag=True)
        pass

    def report_loss(self, epoch, total_loss, train_acc, val_acc):
        self._board_writer.add_scalar('epoch/lr', self._scheduler.get_last_lr()[0], epoch)
        self._board_writer.add_scalar('epoch/loss', total_loss, epoch)
        self._board_writer.add_scalars('epoch/acc', {'train': train_acc, 'val': val_acc}, epoch)
        self._board_writer.flush()
        pass

    def train_epoch(self, epoch):
        train_loader = DataLoader(self._train_dataset, 
                                  batch_size=self._batch_num,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)
        self._train_dataset.update_data()
        self._network = self._network.to(self._device)
        self._network.train()

        with tqdm(total=len(train_loader)) as pbar:
            pbar.set_description(f'Epoch {epoch:02d}')
            for idx, (data, gt) in enumerate(train_loader):
                self._iter_num += 1
                self._optimizer.zero_grad()
                output = self._network(data)
                err = self._loss_func(output, gt)
                err.backward()
                self._avg_meter['loss'].update(err, self._batch_num)
                self._optimizer.step()

                #
                # Compute res
                #
                out_class = torch.argmax(output, dim=1).detach()
                train_acc = metric.multiclass_accuracy(out_class, gt)
                self._avg_meter['train_acc'].update(train_acc, self._batch_num)

                #
                # Reporting
                #
                pbar.update(1)
                if idx % 20 == 0:
                    total_loss = self._avg_meter['loss'].get_iter()
                    train_acc = self._avg_meter['train_acc'].get_iter()
                    for meter_name in self._avg_meter.keys():
                        self._avg_meter[meter_name].clear_iter()
                    self._board_writer.add_scalars('iter/loss-acc', {'loss': total_loss, 'acc': train_acc}, self._iter_num)
                    self._board_writer.flush()
                    pbar.set_postfix(loss=f'{total_loss:.2e}')
            
        # Epoch finished
        total_loss = self._avg_meter['loss'].get_epoch()
        train_acc = self._avg_meter['train_acc'].get_epoch()
        for meter_name in self._avg_meter.keys():
            self._avg_meter[meter_name].clear_epoch()
        return total_loss, train_acc
    
    def test_epoch(self, epoch, vis_flag=False):
        test_loader = DataLoader(self._test_dataset, 
                                 batch_size=self._batch_num,
                                 shuffle=False,
                                 drop_last=False)
        self._network = self._network.to(self._device)
        self._network.eval()

        outputs = []

        with torch.no_grad():
            for idx, (data, gt) in enumerate(test_loader):
                output = self._network(data)
                outputs.append(output)

        # Evaluate
        out_class = torch.argmax(torch.cat(outputs, dim=0), dim=1)
        val_acc = metric.multiclass_accuracy(out_class, self._test_dataset.get_gt(),
                                             num_classes=10, average='macro')
        if self._history_best is None or self._history_best < val_acc:
            self._history_best = val_acc
            self._net_save()
            self._res_save(out_class)
        self.logging(f'Epoch {epoch} end. Eval_acc = {val_acc:.2%}')

        if vis_flag:
            gts = self._test_dataset.get_gt()
            label_names = self._test_dataset.get_label_names()
            val_prec = metric.multiclass_precision(out_class, gts,
                                                   num_classes=10, average='macro')
            val_recl = metric.multiclass_recall(out_class, gts,
                                                num_classes=10, average='macro')
            val_f1 = metric.multiclass_f1_score(out_class, gts,
                                                num_classes=10, average='macro')
            self.logging(f'Report: Acc={val_acc:.2%}, Prd={val_prec:.2%}, Rec:{val_recl:.2%}, F1:{val_f1:.2%}')

            conf_mat = metric.multiclass_confusion_matrix(out_class, gts, num_classes=len(label_names), normalize='pred')
            plt.figure(figsize=(7, 7))
            sbn.heatmap(conf_mat.detach().cpu().numpy(), annot=True, fmt='.1%',
                        xticklabels=label_names, yticklabels=label_names)
            plt.xlabel('Pred')
            plt.ylabel('Truth')
            self._board_writer.add_figure(f'{self._mode}/cm', plt.gcf(), epoch)
            self._board_writer.flush()

        return val_acc
