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
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset

from modelscope.trainers import build_trainer
from modelscope.utils.constant import ModelFile
from modelscope.utils.test_utils import test_level


class DummyMetric:

    def __call__(self, ground_truth, predict_results):
        return {'accuracy': 0.5}


class DummyDataset(Dataset, metaclass=ABCMeta):
    """Base Dataset
    """

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


class TrainerTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_train_0(self):
        json_cfg = {
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
                }, {
                    'type': 'TextLoggerHook',
                    'interval': 1
                }, {
                    'type': 'IterTimerHook'
                }, {
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
                'metrics': ['seq_cls_metric']
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

        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        self.assertIn('epoch_1.pth', results_files)
        self.assertIn('epoch_2.pth', results_files)
        self.assertIn('epoch_3.pth', results_files)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_train_1(self):
        json_cfg = {
            'train': {
                'work_dir':
                self.tmp_dir,
                'dataloader': {
                    'batch_size_per_gpu': 2,
                    'workers_per_gpu': 1
                },
                'hooks': [{
                    'type': 'CheckpointHook',
                    'interval': 1
                }, {
                    'type': 'TextLoggerHook',
                    'interval': 1
                }, {
                    'type': 'IterTimerHook'
                }, {
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
                'metrics': ['seq_cls_metric']
            }
        }

        config_path = os.path.join(self.tmp_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(json_cfg, f)

        model = DummyModel()
        optimmizer = SGD(model.parameters(), lr=0.01)
        lr_scheduler = StepLR(optimmizer, 2)
        trainer_name = 'EpochBasedTrainer'
        kwargs = dict(
            cfg_file=config_path,
            model=model,
            data_collator=None,
            train_dataset=DummyDataset(),
            eval_dataset=DummyDataset(),
            optimizers=(optimmizer, lr_scheduler),
            max_epochs=3)

        trainer = build_trainer(trainer_name, kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)

        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        self.assertIn('epoch_1.pth', results_files)
        self.assertIn('epoch_2.pth', results_files)
        self.assertIn('epoch_3.pth', results_files)


class DummyTrainerTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_dummy(self):
        default_args = dict(cfg_file='configs/examples/train.json')
        trainer = build_trainer('dummy', default_args)

        trainer.train()
        trainer.evaluate()


if __name__ == '__main__':
    unittest.main()
