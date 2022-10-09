# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

import json
import numpy as np
import torch
from torch import nn

from modelscope.metainfo import Trainers
from modelscope.metrics.builder import METRICS, MetricKeys
from modelscope.models.base import Model
from modelscope.trainers import build_trainer
from modelscope.utils.constant import LogKeys, ModelFile
from modelscope.utils.registry import default_group
from modelscope.utils.test_utils import create_dummy_test_dataset

SRC_DIR = os.path.dirname(__file__)


def create_dummy_metric():
    _global_iter = 0

    @METRICS.register_module(
        group_key=default_group, module_name='DummyMetric', force=True)
    class DummyMetric:

        _fake_acc_by_epoch = {1: 0.1, 2: 0.5, 3: 0.2}

        def add(*args, **kwargs):
            pass

        def evaluate(self):
            global _global_iter
            _global_iter += 1
            return {MetricKeys.ACCURACY: self._fake_acc_by_epoch[_global_iter]}


dummy_dataset = create_dummy_test_dataset(
    np.random.random(size=(5, )), np.random.randint(0, 4, (1, )), 20)


class DummyModel(nn.Module, Model):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 4)
        self.bn = nn.BatchNorm1d(4)
        self.model_dir = SRC_DIR

    def forward(self, feat, labels):
        x = self.linear(feat)

        x = self.bn(x)
        loss = torch.sum(x)
        return dict(logits=x, loss=loss)


class CheckpointHookTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        create_dummy_metric()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir)

    def test_checkpoint_hook(self):
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
                    'lr': 0.01,
                    'options': {
                        'grad_clip': {
                            'max_norm': 2.0
                        }
                    }
                },
                'lr_scheduler': {
                    'type': 'StepLR',
                    'step_size': 2,
                    'options': {
                        'warmup': {
                            'type': 'LinearWarmup',
                            'warmup_iters': 2
                        }
                    }
                },
                'hooks': [{
                    'type': 'CheckpointHook',
                    'interval': 1
                }]
            }
        }

        config_path = os.path.join(self.tmp_dir, ModelFile.CONFIGURATION)
        with open(config_path, 'w') as f:
            json.dump(json_cfg, f)

        trainer_name = Trainers.default
        kwargs = dict(
            cfg_file=config_path,
            model=DummyModel(),
            data_collator=None,
            train_dataset=dummy_dataset,
            max_epochs=2)

        trainer = build_trainer(trainer_name, kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{LogKeys.EPOCH}_1.pth', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_2.pth', results_files)

        output_files = os.listdir(
            os.path.join(self.tmp_dir, ModelFile.TRAIN_OUTPUT_DIR))
        self.assertIn(ModelFile.CONFIGURATION, output_files)
        self.assertIn(ModelFile.TORCH_MODEL_BIN_FILE, output_files)
        copy_src_files = os.listdir(SRC_DIR)
        self.assertIn(copy_src_files[0], output_files)
        self.assertIn(copy_src_files[-1], output_files)


class BestCkptSaverHookTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        create_dummy_metric()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir)

    def test_best_checkpoint_hook(self):
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
                    'lr': 0.01
                },
                'lr_scheduler': {
                    'type': 'StepLR',
                    'step_size': 2
                },
                'hooks': [{
                    'type': 'BestCkptSaverHook',
                    'metric_key': MetricKeys.ACCURACY,
                    'rule': 'min'
                }, {
                    'type': 'EvaluationHook',
                    'interval': 1,
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

        trainer_name = Trainers.default
        kwargs = dict(
            cfg_file=config_path,
            model=DummyModel(),
            data_collator=None,
            train_dataset=dummy_dataset,
            eval_dataset=dummy_dataset,
            max_epochs=3)

        trainer = build_trainer(trainer_name, kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'best_{LogKeys.EPOCH}1_{MetricKeys.ACCURACY}0.1.pth',
                      results_files)

        output_files = os.listdir(
            os.path.join(self.tmp_dir, ModelFile.TRAIN_OUTPUT_DIR))
        self.assertIn(ModelFile.CONFIGURATION, output_files)
        self.assertIn(ModelFile.TORCH_MODEL_BIN_FILE, output_files)
        copy_src_files = os.listdir(SRC_DIR)
        self.assertIn(copy_src_files[0], output_files)
        self.assertIn(copy_src_files[-1], output_files)


if __name__ == '__main__':
    unittest.main()
