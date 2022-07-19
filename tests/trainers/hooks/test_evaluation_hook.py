# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest
from abc import ABCMeta

import json
import torch
from torch import nn
from torch.utils.data import Dataset

from modelscope.metrics.builder import METRICS, MetricKeys
from modelscope.trainers import build_trainer
from modelscope.utils.constant import LogKeys, ModelFile
from modelscope.utils.registry import default_group

_global_iter = 0


@METRICS.register_module(group_key=default_group, module_name='DummyMetric')
class DummyMetric:

    _fake_acc_by_epoch = {1: 0.1, 2: 0.5, 3: 0.2}

    def add(*args, **kwargs):
        pass

    def evaluate(self):
        global _global_iter
        _global_iter += 1
        return {MetricKeys.ACCURACY: self._fake_acc_by_epoch[_global_iter]}


class DummyDataset(Dataset, metaclass=ABCMeta):

    def __len__(self):
        return 20

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


class EvaluationHookTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir)

    def test_best_ckpt_rule_max(self):
        global _global_iter
        _global_iter = 0

        json_cfg = {
            'task': 'image_classification',
            'train': {
                'work_dir':
                self.tmp_dir,
                'dataloader': {
                    'batch_size_per_gpu': 2,
                    'workers_per_gpu': 1
                },
                'optimizer': {
                    'type': 'SGD',
                    'lr': 0.01,
                },
                'lr_scheduler': {
                    'type': 'StepLR',
                    'step_size': 2,
                },
                'hooks': [{
                    'type': 'EvaluationHook',
                    'interval': 1,
                    'save_best_ckpt': True,
                    'monitor_key': MetricKeys.ACCURACY
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

        trainer_name = 'EpochBasedTrainer'
        kwargs = dict(
            cfg_file=config_path,
            model=DummyModel(),
            data_collator=None,
            train_dataset=DummyDataset(),
            eval_dataset=DummyDataset(),
            max_epochs=3)

        trainer = build_trainer(trainer_name, kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{LogKeys.EPOCH}_1.pth', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_2.pth', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_3.pth', results_files)
        self.assertIn(f'best_{LogKeys.EPOCH}2_{MetricKeys.ACCURACY}0.5.pth',
                      results_files)

    def test_best_ckpt_rule_min(self):
        global _global_iter
        _global_iter = 0

        json_cfg = {
            'task': 'image_classification',
            'train': {
                'work_dir':
                self.tmp_dir,
                'dataloader': {
                    'batch_size_per_gpu': 2,
                    'workers_per_gpu': 1
                },
                'optimizer': {
                    'type': 'SGD',
                    'lr': 0.01,
                },
                'lr_scheduler': {
                    'type': 'StepLR',
                    'step_size': 2,
                },
                'hooks': [{
                    'type': 'EvaluationHook',
                    'interval': 1,
                    'save_best_ckpt': True,
                    'monitor_key': 'accuracy',
                    'rule': 'min',
                    'out_dir': os.path.join(self.tmp_dir, 'best_ckpt')
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

        trainer_name = 'EpochBasedTrainer'
        kwargs = dict(
            cfg_file=config_path,
            model=DummyModel(),
            data_collator=None,
            train_dataset=DummyDataset(),
            eval_dataset=DummyDataset(),
            max_epochs=3)

        trainer = build_trainer(trainer_name, kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{LogKeys.EPOCH}_1.pth', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_2.pth', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_3.pth', results_files)
        self.assertIn(f'best_{LogKeys.EPOCH}1_{MetricKeys.ACCURACY}0.1.pth',
                      os.listdir(os.path.join(self.tmp_dir, 'best_ckpt')))


if __name__ == '__main__':
    unittest.main()
