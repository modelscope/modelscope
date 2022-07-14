# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest
from abc import ABCMeta

import json
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset

from modelscope.trainers import build_trainer
from modelscope.utils.constant import ModelFile


class DummyDataset(Dataset, metaclass=ABCMeta):
    """Base Dataset
    """

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return dict(feat=torch.rand((5, )), label=torch.randint(0, 4, (1, )))


class DummyModel(nn.Module):

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

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir)

    def test_lr_scheduler_hook(self):
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

        config_path = os.path.join(self.tmp_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(json_cfg, f)

        model = DummyModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        lr_scheduler = MultiStepLR(optimizer, milestones=[2, 4])
        trainer_name = 'EpochBasedTrainer'
        kwargs = dict(
            cfg_file=config_path,
            model=model,
            train_dataset=DummyDataset(),
            optimizers=(optimizer, lr_scheduler),
            max_epochs=5)

        trainer = build_trainer(trainer_name, kwargs)
        train_dataloader = trainer._build_dataloader_with_dataset(
            trainer.train_dataset, **trainer.cfg.train.get('dataloader', {}))
        trainer.register_optimizers_hook()

        trainer.invoke_hook('before_run')
        log_lrs = []
        optim_lrs = []
        for _ in range(trainer._epoch, trainer._max_epochs):
            trainer.invoke_hook('before_train_epoch')
            for _, data_batch in enumerate(train_dataloader):
                trainer.invoke_hook('before_train_iter')

                log_lrs.append(trainer.log_buffer.output['lr'])
                optim_lrs.append(optimizer.param_groups[0]['lr'])

                trainer.train_step(trainer.model, data_batch)
                trainer.invoke_hook('after_train_iter')

            trainer.invoke_hook('after_train_epoch')
            trainer._epoch += 1
        trainer.invoke_hook('after_run')

        iters = 5
        target_lrs = [0.01] * iters * 1 + [0.001] * iters * 2 + [0.0001
                                                                 ] * iters * 2

        self.assertListEqual(log_lrs, target_lrs)
        self.assertListEqual(optim_lrs, target_lrs)

    def test_warmup_lr_scheduler_hook(self):
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
        # optimmizer = SGD(model.parameters(), lr=0.01)
        # lr_scheduler = MultiStepLR(optimmizer, milestones=[2, 4])
        trainer_name = 'EpochBasedTrainer'
        kwargs = dict(
            cfg_file=config_path,
            model=model,
            train_dataset=DummyDataset(),
            # optimizers=(optimmizer, lr_scheduler),
            max_epochs=7)

        trainer = build_trainer(trainer_name, kwargs)
        train_dataloader = trainer._build_dataloader_with_dataset(
            trainer.train_dataset, **trainer.cfg.train.get('dataloader', {}))
        trainer.register_optimizers_hook()

        trainer.invoke_hook('before_run')
        log_lrs = []
        optim_lrs = []
        for _ in range(trainer._epoch, trainer._max_epochs):
            trainer.invoke_hook('before_train_epoch')
            for _, data_batch in enumerate(train_dataloader):
                trainer.invoke_hook('before_train_iter')

                log_lrs.append(round(trainer.log_buffer.output['lr'], 5))
                optim_lrs.append(
                    round(trainer.optimizer.param_groups[0]['lr'], 5))

                trainer.train_step(trainer.model, data_batch)
                trainer.invoke_hook('after_train_iter')

            trainer.invoke_hook('after_train_epoch')
        trainer.invoke_hook('after_run')

        iters = 5
        target_lrs = [0.004] * iters * 1 + [0.007] * iters * 1 + [
            0.01
        ] * iters * 1 + [0.001] * iters * 2 + [0.0001] * iters * 2

        self.assertListEqual(log_lrs, target_lrs)
        self.assertListEqual(optim_lrs, target_lrs)


if __name__ == '__main__':
    unittest.main()
