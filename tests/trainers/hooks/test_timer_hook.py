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


class IterTimerHookTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir)

    def test_iter_time_hook(self):
        json_cfg = {
            'task': 'image_classification',
            'train': {
                'work_dir': self.tmp_dir,
                'dataloader': {
                    'batch_size_per_gpu': 2,
                    'workers_per_gpu': 1
                },
                'hooks': [{
                    'type': 'IterTimerHook',
                }]
            }
        }

        config_path = os.path.join(self.tmp_dir, ModelFile.CONFIGURATION)
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
        trainer.register_hook_from_cfg(trainer.cfg.train.hooks)

        trainer.invoke_hook('before_run')
        for i in range(trainer._epoch, trainer._max_epochs):
            trainer.invoke_hook('before_train_epoch')
            for _, data_batch in enumerate(train_dataloader):
                trainer.invoke_hook('before_train_iter')
                trainer.train_step(trainer.model, data_batch)
                trainer.invoke_hook('after_train_iter')

                self.assertIn('data_load_time', trainer.log_buffer.val_history)
                self.assertIn('time', trainer.log_buffer.val_history)
                self.assertIn('loss', trainer.log_buffer.val_history)

            trainer.invoke_hook('after_train_epoch')

            target_len = 5 * (i + 1)
            self.assertEqual(
                len(trainer.log_buffer.val_history['data_load_time']),
                target_len)
            self.assertEqual(
                len(trainer.log_buffer.val_history['time']), target_len)
            self.assertEqual(
                len(trainer.log_buffer.val_history['loss']), target_len)

            self.assertEqual(
                len(trainer.log_buffer.n_history['data_load_time']),
                target_len)
            self.assertEqual(
                len(trainer.log_buffer.n_history['time']), target_len)
            self.assertEqual(
                len(trainer.log_buffer.n_history['loss']), target_len)

        trainer.invoke_hook('after_run')


if __name__ == '__main__':
    unittest.main()
