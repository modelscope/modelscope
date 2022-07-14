# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime
import os
import os.path as osp
from collections import OrderedDict

import json
import torch
from torch import distributed as dist

from modelscope.utils.torch_utils import get_dist_info
from ..builder import HOOKS
from .base import LoggerHook


@HOOKS.register_module()
class TextLoggerHook(LoggerHook):
    """Logger hook in text, Output log to both console and local json file.

    Args:
        by_epoch (bool, optional): Whether EpochBasedtrainer is used.
            Default: True.
        interval (int, optional): Logging interval (every k iterations).
            Default: 10.
        ignore_last (bool, optional): Ignore the log of last iterations in each
            epoch if less than :attr:`interval`. Default: True.
        reset_flag (bool, optional): Whether to clear the output buffer after
            logging. Default: False.
        out_dir (str): The directory to save log. If is None, use `trainer.work_dir`
    """

    def __init__(self,
                 by_epoch=True,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 out_dir=None):
        super(TextLoggerHook, self).__init__(interval, ignore_last, reset_flag,
                                             by_epoch)
        self.by_epoch = by_epoch
        self.time_sec_tot = 0
        self.out_dir = out_dir
        self._logged_keys = []  # store the key has been logged

    def before_run(self, trainer):
        super(TextLoggerHook, self).before_run(trainer)

        if self.out_dir is None:
            self.out_dir = trainer.work_dir

        if not osp.exists(self.out_dir):
            os.makedirs(self.out_dir)

        trainer.logger.info('Text logs will be saved to {}'.format(
            self.out_dir))

        self.start_iter = trainer.iter
        self.json_log_path = osp.join(self.out_dir,
                                      '{}.log.json'.format(trainer.timestamp))
        if hasattr(trainer, 'meta') and trainer.meta is not None:
            self._dump_log(trainer.meta, trainer)

    def _get_max_memory(self, trainer):
        device = getattr(trainer.model, 'output_device', None)
        mem = torch.cuda.max_memory_allocated(device=device)
        mem_mb = torch.tensor([mem / (1024 * 1024)],
                              dtype=torch.int,
                              device=device)
        _, world_size = get_dist_info()
        if world_size > 1:
            dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
        return mem_mb.item()

    def _log_info(self, log_dict, trainer):
        if log_dict['mode'] == 'train':
            if isinstance(log_dict['lr'], dict):
                lr_str = []
                for k, val in log_dict['lr'].items():
                    lr_str.append(f'lr_{k}: {val:.3e}')
                lr_str = ' '.join(lr_str)
            else:
                lr_str = f'lr: {log_dict["lr"]:.3e}'

            if self.by_epoch:
                log_str = f'Epoch [{log_dict["epoch"]}][{log_dict["iter"]}/{len(trainer.data_loader)}]\t'
            else:
                log_str = f'Iter [{log_dict["iter"]}/{trainer.max_iters}]\t'
            log_str += f'{lr_str}, '
            self._logged_keys.extend(['lr', 'mode', 'iter', 'epoch'])

            if 'time' in log_dict.keys():
                self.time_sec_tot += (log_dict['time'] * self.interval)
                time_sec_avg = self.time_sec_tot / (
                    trainer.iter - self.start_iter + 1)
                eta_sec = time_sec_avg * (trainer.max_iters - trainer.iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str += f'eta: {eta_str}, '
                log_str += f'time: {log_dict["time"]:.3f}, data_load_time: {log_dict["data_load_time"]:.3f}, '
                self._logged_keys.extend([
                    'time',
                    'data_load_time',
                ])
        else:
            # val/test time
            # here 1000 is the length of the val dataloader
            # by epoch: Epoch[val] [4][1000]
            # by iter: Iter[val] [1000]
            if self.by_epoch:
                log_str = f'Epoch({log_dict["mode"]}) [{log_dict["epoch"]}][{log_dict["iter"]}]\t'
            else:
                log_str = f'Iter({log_dict["mode"]}) [{log_dict["iter"]}]\t'
            self._logged_keys.extend(['mode', 'iter', 'epoch'])

        log_items = []
        for name, val in log_dict.items():
            if name in self._logged_keys:
                continue
            if isinstance(val, float):
                val = f'{val:.4f}'
            log_items.append(f'{name}: {val}')
        log_str += ', '.join(log_items)

        trainer.logger.info(log_str)

    def _dump_log(self, log_dict):
        # dump log in json format
        json_log = OrderedDict()
        for k, v in log_dict.items():
            json_log[k] = self._round_float(v)

        rank, _ = get_dist_info()
        if rank == 0:
            with open(self.json_log_path, 'a+') as f:
                json.dump(json_log, f)
                f.write('\n')

    def _round_float(self, items, ndigits=5):
        if isinstance(items, list):
            return [self._round_float(item) for item in items]
        elif isinstance(items, float):
            return round(items, ndigits)
        else:
            return items

    def log(self, trainer):
        cur_iter = self.get_iter(trainer, inner_iter=True)

        log_dict = OrderedDict(
            mode=trainer.mode, epoch=self.get_epoch(trainer), iter=cur_iter)

        # statistic memory
        if torch.cuda.is_available():
            log_dict['memory'] = self._get_max_memory(trainer)

        log_dict = dict(log_dict, **trainer.log_buffer.output)

        self._log_info(log_dict, trainer)
        self._dump_log(log_dict)
        return log_dict
