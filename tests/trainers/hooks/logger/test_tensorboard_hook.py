# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import os
import shutil
import tempfile
import unittest
from abc import ABCMeta

import json
import torch
from torch import nn
from torch.utils.data import Dataset

from modelscope.trainers import build_trainer
from modelscope.utils.constant import LogKeys, ModelFile


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


class TensorboardHookTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir)

    def test_tensorboard_hook(self):
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
                    'type': 'StepLR',
                    'step_size': 2,
                },
                'hooks': [{
                    'type': 'TensorboardHook',
                    'interval': 2
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
            train_dataset=DummyDataset(),
            max_epochs=2)

        trainer = build_trainer(trainer_name, kwargs)
        trainer.train()
        tb_out_dir = os.path.join(self.tmp_dir, 'tensorboard_output')

        events_files = glob.glob(
            os.path.join(tb_out_dir, 'events.out.tfevents.*'))
        self.assertEqual(len(events_files), 1)

        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(events_files[0])
        ea.Reload()
        self.assertEqual(len(ea.Scalars(LogKeys.LOSS)), 10)
        self.assertEqual(len(ea.Scalars(LogKeys.LR)), 10)
        for i in range(5):
            self.assertAlmostEqual(
                ea.Scalars(LogKeys.LR)[i].value, 0.01, delta=0.001)
        for i in range(5, 10):
            self.assertAlmostEqual(
                ea.Scalars(LogKeys.LR)[i].value, 0.001, delta=0.0001)


if __name__ == '__main__':
    unittest.main()
