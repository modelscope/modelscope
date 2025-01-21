# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import os
import shutil
import tempfile
import unittest

import cv2
import json
import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import IterableDataset

from modelscope.metainfo import Metrics, Trainers
from modelscope.metrics.builder import MetricKeys
from modelscope.models.base import TorchModel
from modelscope.trainers import build_trainer
from modelscope.trainers.base import DummyTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.trainer import EpochBasedTrainer
from modelscope.utils.constant import LogKeys, ModeKeys, ModelFile, Tasks
from modelscope.utils.hub import read_config
from modelscope.utils.test_utils import create_dummy_test_dataset, test_level


class DummyIterableDataset(IterableDataset):

    def __iter__(self):
        feat = np.random.random(size=(5, )).astype(np.float32)
        labels = np.random.randint(0, 4, (1, ))
        iterations = [{'feat': feat, 'labels': labels}] * 500
        return iter(iterations)


dummy_dataset_small = create_dummy_test_dataset(
    np.random.random(size=(5, )), np.random.randint(0, 4, (1, )), 20)

dummy_dataset_big = create_dummy_test_dataset(
    np.random.random(size=(5, )), np.random.randint(0, 4, (1, )), 40)


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


@TRAINERS.register_module(module_name='test_vis')
class VisTrainer(EpochBasedTrainer):

    def visualization(self, results, dataset, **kwargs):
        num_image = 5
        f = 'data/test/images/bird.JPEG'
        filenames = [f for _ in range(num_image)]
        imgs = [cv2.imread(f) for f in filenames]
        filenames = [f + str(i) for i in range(num_image)]
        vis_results = {'images': imgs, 'filenames': filenames}

        # visualization results will be displayed in group named eva_vis
        self.visualization_buffer.output['eval_vis'] = vis_results


class TrainerTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_train_0(self):
        json_cfg = {
            'task': Tasks.image_classification,
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
                }, {
                    'type': 'TensorboardHook',
                    'interval': 1
                }]
            },
            'evaluation': {
                'dataloader': {
                    'batch_size_per_gpu': 2,
                    'workers_per_gpu': 1,
                    'shuffle': False
                },
                'metrics': [Metrics.seq_cls_metric],
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
            train_dataset=dummy_dataset_small,
            eval_dataset=dummy_dataset_small,
            max_epochs=3,
            device='cpu')

        trainer = build_trainer(trainer_name, kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)

        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        with open(f'{self.tmp_dir}/{trainer.timestamp}.log', 'r') as infile:
            lines = infile.readlines()
            self.assertTrue(len(lines) > 20)
        self.assertIn(f'{trainer.timestamp}.log', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_1.pth', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_2.pth', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_3.pth', results_files)
        self.assertIn('tensorboard_output', results_files)
        self.assertTrue(len(glob.glob(f'{self.tmp_dir}/*/*events*')) > 0)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_train_visualization(self):
        json_cfg = {
            'task': Tasks.image_classification,
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
                }, {
                    'type': 'TensorboardHook',
                    'interval': 1
                }]
            },
            'evaluation': {
                'dataloader': {
                    'batch_size_per_gpu': 2,
                    'workers_per_gpu': 1,
                    'shuffle': False
                },
                'metrics': [Metrics.seq_cls_metric],
                'visualization': {},
            }
        }
        config_path = os.path.join(self.tmp_dir, ModelFile.CONFIGURATION)
        with open(config_path, 'w') as f:
            json.dump(json_cfg, f)

        trainer_name = 'test_vis'
        kwargs = dict(
            cfg_file=config_path,
            model=DummyModel(),
            data_collator=None,
            train_dataset=dummy_dataset_small,
            eval_dataset=dummy_dataset_small,
            max_epochs=3,
            device='cpu')

        trainer = build_trainer(trainer_name, kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)

        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_1.pth', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_2.pth', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_3.pth', results_files)
        self.assertTrue(len(glob.glob(f'{self.tmp_dir}/*/*events*')) > 0)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_train_1(self):
        json_cfg = {
            'task': Tasks.image_classification,
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
                }, {
                    'type': 'TensorboardHook',
                    'interval': 1
                }]
            },
            'evaluation': {
                'dataloader': {
                    'batch_size_per_gpu': 2,
                    'workers_per_gpu': 1,
                    'shuffle': False
                },
                'metrics': [Metrics.seq_cls_metric]
            }
        }

        config_path = os.path.join(self.tmp_dir, ModelFile.CONFIGURATION)
        with open(config_path, 'w') as f:
            json.dump(json_cfg, f)

        model = DummyModel()
        optimmizer = SGD(model.parameters(), lr=0.01)
        lr_scheduler = StepLR(optimmizer, 2)
        trainer_name = Trainers.default
        kwargs = dict(
            cfg_file=config_path,
            model=model,
            data_collator=None,
            train_dataset=dummy_dataset_small,
            eval_dataset=dummy_dataset_small,
            optimizers=(optimmizer, lr_scheduler),
            max_epochs=3,
            device='cpu')

        trainer = build_trainer(trainer_name, kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)

        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_1.pth', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_2.pth', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_3.pth', results_files)
        self.assertTrue(len(glob.glob(f'{self.tmp_dir}/*/*events*')) > 0)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_train_with_default_config(self):
        json_cfg = {
            'task': Tasks.image_classification,
            'train': {
                'work_dir': self.tmp_dir,
                'dataloader': {
                    'batch_size_per_gpu': 2,
                    'workers_per_gpu': 1
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
                'metrics': [Metrics.seq_cls_metric]
            }
        }

        config_path = os.path.join(self.tmp_dir, ModelFile.CONFIGURATION)
        with open(config_path, 'w') as f:
            json.dump(json_cfg, f)

        model = DummyModel()
        optimmizer = SGD(model.parameters(), lr=0.01)
        lr_scheduler = StepLR(optimmizer, 2)
        trainer_name = Trainers.default
        kwargs = dict(
            cfg_file=config_path,
            model=model,
            data_collator=None,
            train_dataset=dummy_dataset_big,
            eval_dataset=dummy_dataset_small,
            optimizers=(optimmizer, lr_scheduler),
            max_epochs=3,
            device='cpu')

        trainer = build_trainer(trainer_name, kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)

        json_file = os.path.join(self.tmp_dir, f'{trainer.timestamp}.log.json')
        with open(json_file, 'r', encoding='utf-8') as f:
            lines = [i.strip() for i in f.readlines()]
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.TRAIN,
                LogKeys.EPOCH: 1,
                LogKeys.ITER: 10,
                LogKeys.LR: 0.01
            }, json.loads(lines[0]))
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.TRAIN,
                LogKeys.EPOCH: 1,
                LogKeys.ITER: 20,
                LogKeys.LR: 0.01
            }, json.loads(lines[1]))
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.EVAL,
                LogKeys.EPOCH: 1,
                LogKeys.ITER: 10
            }, json.loads(lines[2]))
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.TRAIN,
                LogKeys.EPOCH: 2,
                LogKeys.ITER: 10,
                LogKeys.LR: 0.01
            }, json.loads(lines[3]))
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.TRAIN,
                LogKeys.EPOCH: 2,
                LogKeys.ITER: 20,
                LogKeys.LR: 0.01
            }, json.loads(lines[4]))
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.EVAL,
                LogKeys.EPOCH: 2,
                LogKeys.ITER: 10
            }, json.loads(lines[5]))
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.TRAIN,
                LogKeys.EPOCH: 3,
                LogKeys.ITER: 10,
                LogKeys.LR: 0.001
            }, json.loads(lines[6]))
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.TRAIN,
                LogKeys.EPOCH: 3,
                LogKeys.ITER: 20,
                LogKeys.LR: 0.001
            }, json.loads(lines[7]))
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.EVAL,
                LogKeys.EPOCH: 3,
                LogKeys.ITER: 10
            }, json.loads(lines[8]))
        self.assertIn(f'{LogKeys.EPOCH}_1.pth', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_2.pth', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_3.pth', results_files)
        for i in [0, 1, 3, 4, 6, 7]:
            self.assertIn(LogKeys.DATA_LOAD_TIME, lines[i])
            self.assertIn(LogKeys.ITER_TIME, lines[i])
        for i in [2, 5, 8]:
            self.assertIn(MetricKeys.ACCURACY, lines[i])

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_train_with_iters_per_epoch(self):
        json_cfg = {
            'task': Tasks.image_classification,
            'train': {
                'work_dir': self.tmp_dir,
                'dataloader': {
                    'batch_size_per_gpu': 2,
                    'workers_per_gpu': 1
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
                'metrics': [Metrics.seq_cls_metric]
            }
        }
        config_path = os.path.join(self.tmp_dir, ModelFile.CONFIGURATION)
        with open(config_path, 'w') as f:
            json.dump(json_cfg, f)

        model = DummyModel()
        optimmizer = SGD(model.parameters(), lr=0.01)
        lr_scheduler = StepLR(optimmizer, 2)
        trainer_name = Trainers.default
        kwargs = dict(
            cfg_file=config_path,
            model=model,
            data_collator=None,
            optimizers=(optimmizer, lr_scheduler),
            train_dataset=DummyIterableDataset(),
            eval_dataset=DummyIterableDataset(),
            train_iters_per_epoch=20,
            val_iters_per_epoch=10,
            max_epochs=3,
            device='cpu')

        trainer = build_trainer(trainer_name, kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        json_file = os.path.join(self.tmp_dir, f'{trainer.timestamp}.log.json')
        with open(json_file, 'r', encoding='utf-8') as f:
            lines = [i.strip() for i in f.readlines()]
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.TRAIN,
                LogKeys.EPOCH: 1,
                LogKeys.ITER: 10,
                LogKeys.LR: 0.01
            }, json.loads(lines[0]))
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.TRAIN,
                LogKeys.EPOCH: 1,
                LogKeys.ITER: 20,
                LogKeys.LR: 0.01
            }, json.loads(lines[1]))
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.EVAL,
                LogKeys.EPOCH: 1,
                LogKeys.ITER: 10
            }, json.loads(lines[2]))
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.TRAIN,
                LogKeys.EPOCH: 2,
                LogKeys.ITER: 10,
                LogKeys.LR: 0.01
            }, json.loads(lines[3]))
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.TRAIN,
                LogKeys.EPOCH: 2,
                LogKeys.ITER: 20,
                LogKeys.LR: 0.01
            }, json.loads(lines[4]))
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.EVAL,
                LogKeys.EPOCH: 2,
                LogKeys.ITER: 10
            }, json.loads(lines[5]))
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.TRAIN,
                LogKeys.EPOCH: 3,
                LogKeys.ITER: 10,
                LogKeys.LR: 0.001
            }, json.loads(lines[6]))
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.TRAIN,
                LogKeys.EPOCH: 3,
                LogKeys.ITER: 20,
                LogKeys.LR: 0.001
            }, json.loads(lines[7]))
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.EVAL,
                LogKeys.EPOCH: 3,
                LogKeys.ITER: 10
            }, json.loads(lines[8]))
        self.assertIn(f'{LogKeys.EPOCH}_1.pth', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_2.pth', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_3.pth', results_files)
        for i in [0, 1, 3, 4, 6, 7]:
            self.assertIn(LogKeys.DATA_LOAD_TIME, lines[i])
            self.assertIn(LogKeys.ITER_TIME, lines[i])
        for i in [2, 5, 8]:
            self.assertIn(MetricKeys.ACCURACY, lines[i])

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_train_with_old_and_new_cfg(self):
        old_cfg = {
            'task': Tasks.image_classification,
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
                }, {
                    'type': 'TensorboardHook',
                    'interval': 1
                }]
            },
            'evaluation': {
                'dataloader': {
                    'batch_size_per_gpu': 2,
                    'workers_per_gpu': 1,
                    'shuffle': False
                },
                'metrics': [Metrics.seq_cls_metric],
            }
        }

        new_cfg = {
            'task': Tasks.image_classification,
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
                'checkpoint': {
                    'period': {
                        'interval': 1
                    }
                },
                'logging': {
                    'interval': 1
                },
                'hooks': [{
                    'type': 'IterTimerHook'
                }, {
                    'type': 'TensorboardHook',
                    'interval': 1
                }]
            },
            'evaluation': {
                'dataloader': {
                    'batch_size_per_gpu': 2,
                    'workers_per_gpu': 1,
                    'shuffle': False
                },
                'metrics': [Metrics.seq_cls_metric],
                'period': {
                    'interval': 1
                }
            }
        }

        def assert_new_cfg(cfg):
            self.assertNotIn('CheckpointHook', cfg.train.hooks)
            self.assertNotIn('TextLoggerHook', cfg.train.hooks)
            self.assertNotIn('EvaluationHook', cfg.train.hooks)
            self.assertIn('checkpoint', cfg.train)
            self.assertIn('logging', cfg.train)
            self.assertIn('period', cfg.evaluation)

        for json_cfg in (new_cfg, old_cfg):
            config_path = os.path.join(self.tmp_dir, ModelFile.CONFIGURATION)
            with open(config_path, 'w') as f:
                json.dump(json_cfg, f)
            trainer_name = Trainers.default
            kwargs = dict(
                cfg_file=config_path,
                model=DummyModel(),
                data_collator=None,
                train_dataset=dummy_dataset_small,
                eval_dataset=dummy_dataset_small,
                max_epochs=3,
                device='cpu')

            trainer = build_trainer(trainer_name, kwargs)
            assert_new_cfg(trainer.cfg)
            trainer.train()
            cfg = read_config(os.path.join(self.tmp_dir, 'output'))
            assert_new_cfg(cfg)


class DummyTrainerTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_dummy(self):
        default_args = dict(cfg_file='configs/examples/train.json')
        trainer = build_trainer('dummy', default_args)

        trainer.train()
        trainer.evaluate()


if __name__ == '__main__':
    unittest.main()
