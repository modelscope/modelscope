# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

import json
import numpy as np
import torch
from torch import nn

from modelscope.trainers import build_trainer
from modelscope.utils.constant import LogKeys, ModelFile
from modelscope.utils.test_utils import create_dummy_test_dataset

dummy_dataset = create_dummy_test_dataset(
    np.random.random(size=(5, )), np.random.randint(0, 4, (1, )), 20)


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


class CheckpointHookTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir)

    def test_checkpoint_hook(self):
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

        trainer_name = 'EpochBasedTrainer'
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


if __name__ == '__main__':
    unittest.main()
