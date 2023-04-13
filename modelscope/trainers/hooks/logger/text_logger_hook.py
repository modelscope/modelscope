# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime
import os
import os.path as osp
from collections import OrderedDict

import json
import torch
from torch import distributed as dist

from modelscope.metainfo import Hooks
from modelscope.trainers.hooks.builder import HOOKS
from modelscope.utils.constant import LogKeys, ModeKeys
from modelscope.utils.json_utils import EnhancedEncoder
from modelscope.utils.torch_utils import is_master
from .base import LoggerHook


@HOOKS.register_module(module_name=Hooks.TextLoggerHook)
class TextLoggerHook(LoggerHook):
    """Logger hook in text, Output log to both console and local json file.

    Args:
        by_epoch (bool, optional): Whether EpochBasedTrainer is used.
            Default: True.
        interval (int, optional): Logging interval (every k iterations).
            It is interval of iterations even by_epoch is true. Default: 10.
        ignore_last (bool, optional): Ignore the log of last iterations in each
            epoch if less than :attr:`interval`. Default: True.
        reset_flag (bool, optional): Whether to clear the output buffer after
            logging. Default: False.
        out_dir (str): The directory to save log. If is None, use `trainer.work_dir`
        ignore_rounding_keys (`Union[str, List]`): The keys to ignore float rounding, default 'lr'
        rounding_digits (`int`): The digits of rounding, exceeding parts will be ignored.
    """

    def __init__(self,
                 by_epoch=True,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 out_dir=None,
                 ignore_rounding_keys='lr',
                 rounding_digits=5):
        super(TextLoggerHook, self).__init__(interval, ignore_last, reset_flag,
                                             by_epoch)
        self.by_epoch = by_epoch
        self.time_sec_tot = 0
        self.out_dir = out_dir
        self._logged_keys = []  # store the key has been logged
        if isinstance(ignore_rounding_keys,
                      str) or ignore_rounding_keys is None:
            ignore_rounding_keys = [ignore_rounding_keys]
        self.ignore_rounding_keys = ignore_rounding_keys
        self.rounding_digits = rounding_digits

    def before_run(self, trainer):
        super(TextLoggerHook, self).before_run(trainer)

        if self.out_dir is None:
            self.out_dir = trainer.work_dir

        if not osp.exists(self.out_dir) and is_master():
            os.makedirs(self.out_dir)

        trainer.logger.info('Text logs will be saved to {}'.format(
            self.out_dir))

        self.start_iter = trainer.iter
        self.json_log_path = osp.join(self.out_dir,
                                      '{}.log.json'.format(trainer.timestamp))
        if hasattr(trainer, 'meta') and trainer.meta is not None:
            self._dump_log(trainer.meta)

    def _get_max_memory(self, trainer):
        device = torch.cuda.current_device()
        mem = torch.cuda.max_memory_allocated(device=device)
        mem_mb = torch.tensor([int(mem / (1024 * 1024))],
                              dtype=torch.int,
                              device=device)
        if trainer._dist:
            dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
        return mem_mb.item()

    def _log_info(self, log_dict, trainer):
        lr_key = LogKeys.LR
        epoch_key = LogKeys.EPOCH
        iter_key = LogKeys.ITER
        mode_key = LogKeys.MODE
        iter_time_key = LogKeys.ITER_TIME
        data_load_time_key = LogKeys.DATA_LOAD_TIME
        eta_key = LogKeys.ETA

        if log_dict[mode_key] == ModeKeys.TRAIN:
            if isinstance(log_dict[lr_key], dict):
                lr_str = []
                for k, val in log_dict[lr_key].items():
                    lr_str.append(f'{lr_key}_{k}: {val:.3e}')
                lr_str = ' '.join(lr_str)
            else:
                lr_str = f'{lr_key}: {log_dict[lr_key]:.3e}'

            if self.by_epoch:
                log_str = f'{epoch_key} [{log_dict[epoch_key]}][{log_dict[iter_key]}/{trainer.iters_per_epoch}]\t'
            else:
                log_str = f'{iter_key} [{log_dict[iter_key]}/{trainer.max_iters}]\t'
            log_str += f'{lr_str}, '
            self._logged_keys.extend([lr_key, mode_key, iter_key, epoch_key])

            if iter_time_key in log_dict.keys():
                self.time_sec_tot += (log_dict[iter_time_key] * self.interval)
                time_sec_avg = self.time_sec_tot / (
                    trainer.iter - self.start_iter + 1)
                eta_sec = time_sec_avg * (trainer.max_iters - trainer.iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str += f'{eta_key}: {eta_str}, '
                log_str += f'{iter_time_key}: {log_dict[iter_time_key]:.3f}, '
                log_str += f'{data_load_time_key}: {log_dict[data_load_time_key]:.3f}, '
                self._logged_keys.extend([
                    iter_time_key,
                    data_load_time_key,
                ])
        else:
            # val/test time
            # here 1000 is the length of the val dataloader
            # by epoch: epoch[val] [4][1000]
            # by iter: iter[val] [1000]
            if self.by_epoch:
                log_str = f'{epoch_key}({log_dict[mode_key]}) [{log_dict[epoch_key]}][{log_dict[iter_key]}]\t'
            else:
                # TODO log_dict[iter_key] is not correct because of it's train_loop's inner_iter
                log_str = f'{iter_key}({log_dict[mode_key]}) [{log_dict[iter_key]}]\t'
            self._logged_keys.extend([mode_key, iter_key, epoch_key])

        log_items = []
        for name, val in log_dict.items():
            if name in self._logged_keys:
                continue
            if isinstance(val,
                          float) and name not in self.ignore_rounding_keys:
                val = f'{val:.4f}'
            log_items.append(f'{name}: {val}')
        log_str += ', '.join(log_items)

        if is_master():
            trainer.logger.info(log_str)

    def _dump_log(self, log_dict):
        # dump log in json format
        json_log = OrderedDict()
        for k, v in log_dict.items():
            json_log[
                k] = v if k in self.ignore_rounding_keys else self._round_float(
                    v, self.rounding_digits)

        if is_master():
            with open(self.json_log_path, 'a+') as f:
                json.dump(json_log, f, cls=EnhancedEncoder)
                f.write('\n')

    def _round_float(self, items, ndigits=5):
        if isinstance(items, list):
            return [self._round_float(item, ndigits) for item in items]
        elif isinstance(items, float):
            return round(items, ndigits)
        else:
            return items

    def log(self, trainer):
        cur_iter = self.get_iter(
            trainer, inner_iter=True
        ) if trainer.mode == ModeKeys.TRAIN else trainer.iters_per_epoch

        log_dict = OrderedDict(
            mode=trainer.mode, epoch=self.get_epoch(trainer), iter=cur_iter)

        # statistic memory
        if torch.cuda.is_available():
            log_dict[LogKeys.MEMORY] = self._get_max_memory(trainer)

        log_dict = dict(log_dict, **trainer.log_buffer.output)

        self._log_info(log_dict, trainer)
        self._dump_log(log_dict)
        return log_dict
