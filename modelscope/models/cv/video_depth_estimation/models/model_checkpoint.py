# The implementation is adopted from Pytorch-Lightning
# https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/callbacks/model_checkpoint.py

import os
import re

import numpy as np
import torch


def save_code(filepath):
    """Save code in the models folder"""
    os.system('tar cfz {}/code.tar.gz *'.format(filepath))


class ModelCheckpoint:

    def __init__(self,
                 filepath=None,
                 monitor='val_loss',
                 save_top_k=1,
                 mode='auto',
                 period=1,
                 s3_path='',
                 s3_frequency=5):
        super().__init__()
        # If save_top_k is zero, save all models
        if save_top_k == 0:
            save_top_k = 1e6
        # Create checkpoint folder
        self.dirpath, self.filename = os.path.split(filepath)
        print(self.dirpath, self.filename, filepath)
        os.makedirs(self.dirpath, exist_ok=True)
        # Store arguments
        self.monitor = monitor
        self.save_top_k = save_top_k
        self.period = period
        self.epoch_last_check = None
        self.best_k_models = {}
        self.kth_best_model = ''
        self.best = 0
        # Monitoring modes
        torch_inf = torch.tensor(np.Inf)
        mode_dict = {
            'min': (torch_inf, 'min'),
            'max': (-torch_inf, 'max'),
            'auto': (-torch_inf, 'max') if 'acc' in self.monitor
            or 'a1' in self.monitor or self.monitor.startswith('fmeasure') else
            (torch_inf, 'min'),
        }
        self.kth_value, self.mode = mode_dict[mode]

        self.s3_path = s3_path
        self.s3_frequency = s3_frequency
        self.s3_enabled = (s3_path != '') and (s3_frequency > 0)
        self.save_code = True

    @staticmethod
    def _del_model(filepath):
        if os.path.isfile(filepath):
            os.remove(filepath)

    def _save_model(self, filepath, model):
        # Create folder, save model and sync to s3
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(
            {
                'config': model.config,
                'epoch': model.current_epoch,
                'state_dict': model.state_dict(),
                'optimizer': model.optimizer.state_dict(),
                'scheduler': model.scheduler.state_dict(),
            }, filepath)

    def check_monitor_top_k(self, current):
        # If we don't have enough models
        if len(self.best_k_models) < self.save_top_k:
            return True
        # Convert to torch if necessary
        if not isinstance(current, torch.Tensor):
            current = torch.tensor(current)
        # Get monitoring operation
        monitor_op = {
            'min': torch.lt,
            'max': torch.gt,
        }[self.mode]
        # Compare and return
        return monitor_op(current, self.best_k_models[self.kth_best_model])

    def format_checkpoint_name(self, epoch, metrics):
        metrics['epoch'] = epoch
        filename = self.filename
        for tmp in re.findall(r'(\{.*?)[:\}]', self.filename):
            name = tmp[1:]
            filename = filename.replace(tmp, name + '={' + name)
            if name not in metrics:
                metrics[name] = 0
        filename = filename.format(**metrics)
        return os.path.join(self.dirpath, '{}.ckpt'.format(filename))

    def check_and_save(self, model, metrics):
        # Check saving interval
        epoch = model.current_epoch
        if self.epoch_last_check is not None and \
                (epoch - self.epoch_last_check) < self.period:
            return
        self.epoch_last_check = epoch
        # Prepare filepath
        filepath = self.format_checkpoint_name(epoch, metrics)
        while os.path.isfile(filepath):
            filepath = self.format_checkpoint_name(epoch, metrics)
        # Check if saving or not
        if self.save_top_k != -1:
            current = metrics.get(self.monitor)
            assert current, 'Checkpoint metric is not available'
            if self.check_monitor_top_k(current):
                self._do_check_save(filepath, model, current)
        else:
            self._save_model(filepath, model)

    def _do_check_save(self, filepath, model, current):
        # List of models to delete
        del_list = []
        if len(self.best_k_models) == self.save_top_k and self.save_top_k > 0:
            delpath = self.kth_best_model
            self.best_k_models.pop(self.kth_best_model)
            del_list.append(delpath)
        # Monitor current models
        self.best_k_models[filepath] = current
        if len(self.best_k_models) == self.save_top_k:
            # Monitor dict has reached k elements
            _op = max if self.mode == 'min' else min
            self.kth_best_model = _op(
                self.best_k_models, key=self.best_k_models.get)
            self.kth_value = self.best_k_models[self.kth_best_model]
        # Determine best model
        _op = min if self.mode == 'min' else max
        self.best = _op(self.best_k_models.values())
        # Delete old models
        for cur_path in del_list:
            if cur_path != filepath:
                self._del_model(cur_path)
        # Save model
        self._save_model(filepath, model)
