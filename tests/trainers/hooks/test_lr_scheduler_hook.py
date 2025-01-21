# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

import json
import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR, MultiStepLR

from modelscope.metainfo import Trainers
from modelscope.metrics.builder import METRICS, MetricKeys
from modelscope.models.base import TorchModel
from modelscope.trainers import build_trainer
from modelscope.trainers.default_config import merge_hooks
from modelscope.utils.constant import LogKeys, ModelFile, TrainerStages
from modelscope.utils.registry import default_group
from modelscope.utils.test_utils import create_dummy_test_dataset

dummy_dataset = create_dummy_test_dataset(
    np.random.random(size=(5, )), np.random.randint(0, 4, (1, )), 10)


def create_dummy_metric():
    _global_iter = 0

    @METRICS.register_module(
        group_key=default_group, module_name='DummyMetric', force=True)
    class DummyMetric:

        _fake_acc_by_epoch = {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.3}

        def add(*args, **kwargs):
            pass

        def evaluate(self):
            global _global_iter
            _global_iter += 1
            return {MetricKeys.ACCURACY: self._fake_acc_by_epoch[_global_iter]}


class DummyModel(TorchModel):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 4)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, feat, labels):
        x = self.linear(feat)

        x = self.bn(x)
        loss = torch.sum(x)
        return dict(logits=x, loss=loss)


class LrSchedulerHookTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        create_dummy_metric()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir)

    def test_lr_scheduler_hook(self):
        global _global_iter
        _global_iter = 0

        json_cfg = {
            'task': 'image_classification',
            'train': {
                'work_dir': self.tmp_dir,
                'dataloader': {
                    'batch_size_per_gpu': 2,
                    'workers_per_gpu': 1
                }
            }
        }

        config_path = os.path.join(self.tmp_dir, ModelFile.CONFIGURATION)
        with open(config_path, 'w') as f:
            json.dump(json_cfg, f)

        model = DummyModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        lr_scheduler = MultiStepLR(optimizer, milestones=[2, 4])
        trainer_name = Trainers.default
        kwargs = dict(
            cfg_file=config_path,
            model=model,
            train_dataset=dummy_dataset,
            optimizers=(optimizer, lr_scheduler),
            max_epochs=5,
            device='cpu')

        trainer = build_trainer(trainer_name, kwargs)
        train_dataloader = trainer._build_dataloader_with_dataset(
            trainer.train_dataset, **trainer.cfg.train.get('dataloader', {}))
        trainer.register_optimizers_hook()
        trainer.register_processors()
        trainer._hooks = [
            hook for hook in trainer._hooks if hook.__class__.__name__ not in
            ['CheckpointHook', 'TextLoggerHook', 'IterTimerHook']
        ]
        trainer.invoke_hook(TrainerStages.before_run)
        log_lrs = []
        optim_lrs = []
        for _ in range(trainer._epoch, trainer._max_epochs):
            trainer.invoke_hook(TrainerStages.before_train_epoch)
            for _, data_batch in enumerate(train_dataloader):
                trainer.invoke_hook(TrainerStages.before_train_iter)
                trainer.train_step(trainer.model, data_batch)
                trainer.invoke_hook(TrainerStages.after_train_iter)

                log_lrs.append(trainer.log_buffer.output[LogKeys.LR])
                optim_lrs.append(optimizer.param_groups[0]['lr'])

            trainer.invoke_hook(TrainerStages.after_train_epoch)
            trainer._epoch += 1
        trainer.invoke_hook(TrainerStages.after_run)

        iters = 5
        target_lrs = [0.01] * iters * 2 + [0.001] * iters * 2 + [0.0001
                                                                 ] * iters * 1
        self.assertListEqual(log_lrs, target_lrs)
        self.assertListEqual(optim_lrs, target_lrs)

    def test_accumulation_step(self):
        json_cfg = {
            'task': 'image_classification',
            'train': {
                'work_dir': self.tmp_dir,
                'dataloader': {
                    'batch_size_per_gpu': 2,
                    'workers_per_gpu': 1
                },
                'optimizer': {
                    'type': 'SGD',
                    'lr': 0.01,
                    'options': {
                        'cumulative_iters': 4,
                    }
                },
                'lr_scheduler': {
                    'type': 'LinearLR',
                    'start_factor': 1.0,
                    'end_factor': 0.0,
                    'total_iters': int(8 * len(dummy_dataset) / 2),
                    'options': {
                        'by_epoch': False,
                    }
                }
            }
        }

        config_path = os.path.join(self.tmp_dir, ModelFile.CONFIGURATION)
        with open(config_path, 'w') as f:
            json.dump(json_cfg, f)

        model = DummyModel()
        trainer_name = Trainers.default
        kwargs = dict(
            cfg_file=config_path,
            model=model,
            train_dataset=dummy_dataset,
            max_epochs=8,
            device='cpu')

        trainer = build_trainer(trainer_name, kwargs)
        train_dataloader = trainer._build_dataloader_with_dataset(
            trainer.train_dataset, **trainer.cfg.train.get('dataloader', {}))
        trainer.register_optimizers_hook()
        trainer.register_processors()
        trainer._hooks = [
            hook for hook in trainer._hooks if hook.__class__.__name__ not in
            ['CheckpointHook', 'TextLoggerHook', 'IterTimerHook']
        ]
        trainer.invoke_hook(TrainerStages.before_run)
        log_lrs = []
        optim_lrs = []
        for epoch in range(trainer._epoch, trainer._max_epochs):
            trainer.invoke_hook(TrainerStages.before_train_epoch)
            for iter, data_batch in enumerate(train_dataloader):
                trainer.invoke_hook(TrainerStages.before_train_iter)
                trainer.train_step(trainer.model, data_batch)
                trainer.invoke_hook(TrainerStages.after_train_iter)

                if (trainer.iter + 1) % 4 == 0:
                    log_lrs.append(trainer.log_buffer.output[LogKeys.LR])
                    optim_lrs.append(trainer.optimizer.param_groups[0]['lr'])

                trainer._iter += 1

            trainer.invoke_hook(TrainerStages.after_train_epoch)
            trainer._epoch += 1
        trainer.invoke_hook(TrainerStages.after_run)
        lr = 0.01
        decay = 0.01 / 40
        target_lrs = []
        for i in range(40):
            if i >= 3:
                lr -= decay
                target_lrs.append(lr)
            else:
                target_lrs.append(lr)
        target_lrs = [
            i for idx, i in enumerate(target_lrs) if (idx + 1) % 4 == 0
        ]
        self.assertTrue(all(np.isclose(log_lrs, target_lrs)))
        self.assertTrue(all(np.isclose(optim_lrs, target_lrs)))

    def test_warmup_lr_scheduler_hook(self):
        global _global_iter
        _global_iter = 0

        json_cfg = {
            'task': 'image_classification',
            'train': {
                'work_dir': self.tmp_dir,
                'dataloader': {
                    'batch_size_per_gpu': 2,
                    'workers_per_gpu': 1
                },
                'optimizer': {
                    'type': 'SGD',
                    'lr': 0.01
                },
                'lr_scheduler': {
                    'type': 'MultiStepLR',
                    'milestones': [4, 6],
                    'options': {
                        'warmup': {
                            'type': 'LinearWarmup',
                            'warmup_iters': 3
                        }
                    }
                }
            }
        }

        config_path = os.path.join(self.tmp_dir, ModelFile.CONFIGURATION)
        with open(config_path, 'w') as f:
            json.dump(json_cfg, f)

        model = DummyModel()
        trainer_name = Trainers.default
        kwargs = dict(
            cfg_file=config_path,
            model=model,
            train_dataset=dummy_dataset,
            max_epochs=7,
            device='cpu')

        trainer = build_trainer(trainer_name, kwargs)
        train_dataloader = trainer._build_dataloader_with_dataset(
            trainer.train_dataset, **trainer.cfg.train.get('dataloader', {}))
        trainer.register_optimizers_hook()
        trainer._hooks = [
            hook for hook in trainer._hooks if hook.__class__.__name__ not in
            ['CheckpointHook', 'TextLoggerHook', 'IterTimerHook']
        ]
        trainer.invoke_hook(TrainerStages.before_run)
        log_lrs = []
        optim_lrs = []
        for _ in range(trainer._epoch, trainer._max_epochs):
            trainer.invoke_hook(TrainerStages.before_train_epoch)
            for _, data_batch in enumerate(train_dataloader):
                trainer.invoke_hook(TrainerStages.before_train_iter)
                trainer.train_step(trainer.model, data_batch)
                trainer.invoke_hook(TrainerStages.after_train_iter)

                log_lrs.append(round(trainer.log_buffer.output[LogKeys.LR], 5))
                optim_lrs.append(
                    round(trainer.optimizer.param_groups[0]['lr'], 5))

            trainer.invoke_hook(TrainerStages.after_train_epoch)
        trainer.invoke_hook(TrainerStages.after_run)

        iters = 5
        target_lrs = [0.001] * iters * 1 + [0.004] * iters * 1 + [
            0.007
        ] * iters * 1 + [0.01] * iters * 1 + [0.001] * iters * 2 + [
            0.0001
        ] * iters * 1

        self.assertListEqual(log_lrs, target_lrs)
        self.assertListEqual(optim_lrs, target_lrs)


class PlateauLrSchedulerHookTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        create_dummy_metric()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir)

    def test_plateau_lr_scheduler_hook(self):
        global _global_iter
        _global_iter = 0

        json_cfg = {
            'task': 'image_classification',
            'train': {
                'work_dir': self.tmp_dir,
                'dataloader': {
                    'batch_size_per_gpu': 2,
                    'workers_per_gpu': 1
                },
                'lr_scheduler': {
                    'type': 'ReduceLROnPlateau',
                    'mode': 'max',
                    'factor': 0.1,
                    'patience': 2,
                },
                'lr_scheduler_hook': {
                    'type': 'PlateauLrSchedulerHook',
                    'metric_key': MetricKeys.ACCURACY
                },
                'hooks': [{
                    'type': 'EvaluationHook',
                    'interval': 1
                }]
            },
            'evaluation': {
                'dataloader': {
                    'batch_size_per_gpu': 2,
                    'workers_per_gpu': 1,
                    'shuffle': False
                },
                'metrics': ['DummyMetric']
            }
        }

        config_path = os.path.join(self.tmp_dir, ModelFile.CONFIGURATION)
        with open(config_path, 'w') as f:
            json.dump(json_cfg, f)

        model = DummyModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        trainer_name = Trainers.default
        kwargs = dict(
            cfg_file=config_path,
            model=model,
            train_dataset=dummy_dataset,
            eval_dataset=dummy_dataset,
            optimizers=(optimizer, None),
            max_epochs=5,
            device='cpu')

        trainer = build_trainer(trainer_name, kwargs)
        train_dataloader = trainer._build_dataloader_with_dataset(
            trainer.train_dataset, **trainer.cfg.train.get('dataloader', {}))
        trainer.train_dataloader = train_dataloader
        trainer.data_loader = train_dataloader
        trainer.register_optimizers_hook()
        trainer.register_processors()
        trainer._hooks = [
            hook for hook in trainer._hooks if hook.__class__.__name__ not in
            ['CheckpointHook', 'TextLoggerHook', 'IterTimerHook']
        ]
        trainer.invoke_hook(TrainerStages.before_run)
        log_lrs = []
        optim_lrs = []
        for _ in range(trainer._epoch, trainer._max_epochs):
            trainer.invoke_hook(TrainerStages.before_train_epoch)
            for _, data_batch in enumerate(train_dataloader):
                trainer.invoke_hook(TrainerStages.before_train_iter)
                trainer.train_step(trainer.model, data_batch)
                trainer.invoke_hook(TrainerStages.after_train_iter)

                log_lrs.append(trainer.log_buffer.output[LogKeys.LR])
                optim_lrs.append(optimizer.param_groups[0]['lr'])

            trainer.invoke_hook(TrainerStages.after_train_epoch)
            trainer._epoch += 1
        trainer.invoke_hook(TrainerStages.after_run)

        iters = 5
        target_lrs = [0.01] * iters * 4 + [0.001] * iters * 1
        self.assertListEqual(log_lrs, target_lrs)
        self.assertListEqual(optim_lrs, target_lrs)


if __name__ == '__main__':
    unittest.main()
