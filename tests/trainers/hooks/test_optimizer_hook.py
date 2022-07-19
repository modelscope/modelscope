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
from modelscope.utils.constant import ModelFile, TrainerStages


class DummyDataset(Dataset, metaclass=ABCMeta):
    """Base Dataset
    """

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return dict(feat=torch.rand((2, 2)), label=torch.randint(0, 2, (1, )))


class DummyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.bn = nn.BatchNorm1d(2)

    def forward(self, feat, labels):
        x = self.linear(feat)
        x = self.bn(x)
        loss = torch.sum(x)
        return dict(logits=x, loss=loss)


class OptimizerHookTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir)

    def test_optimizer_hook(self):
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
        lr_scheduler = MultiStepLR(optimizer, milestones=[1, 2])
        trainer_name = 'EpochBasedTrainer'
        kwargs = dict(
            cfg_file=config_path,
            model=model,
            train_dataset=DummyDataset(),
            optimizers=(optimizer, lr_scheduler),
            max_epochs=2)

        trainer = build_trainer(trainer_name, kwargs)
        train_dataloader = trainer._build_dataloader_with_dataset(
            trainer.train_dataset, **trainer.cfg.train.get('dataloader', {}))
        trainer.register_optimizers_hook()

        trainer.invoke_hook(TrainerStages.before_run)

        for _ in range(trainer._epoch, trainer._max_epochs):
            trainer.invoke_hook(TrainerStages.before_train_epoch)
            for _, data_batch in enumerate(train_dataloader):
                trainer.invoke_hook(TrainerStages.before_train_iter)
                trainer.train_step(trainer.model, data_batch)
                trainer.invoke_hook(TrainerStages.after_train_iter)

                self.assertEqual(
                    len(trainer.optimizer.param_groups[0]['params']), 4)
                for i in range(4):
                    self.assertTrue(trainer.optimizer.param_groups[0]['params']
                                    [i].requires_grad)

            trainer.invoke_hook(TrainerStages.after_train_epoch)
            trainer._epoch += 1
        trainer.invoke_hook(TrainerStages.after_run)


class TorchAMPOptimizerHookTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir)

    def test_amp_optimizer_hook(self):
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

        model = DummyModel().cuda()
        optimizer = SGD(model.parameters(), lr=0.01)
        lr_scheduler = MultiStepLR(optimizer, milestones=[1, 2])
        trainer_name = 'EpochBasedTrainer'
        kwargs = dict(
            cfg_file=config_path,
            model=model,
            train_dataset=DummyDataset(),
            optimizers=(optimizer, lr_scheduler),
            max_epochs=2,
            use_fp16=True)

        trainer = build_trainer(trainer_name, kwargs)
        train_dataloader = trainer._build_dataloader_with_dataset(
            trainer.train_dataset, **trainer.cfg.train.get('dataloader', {}))
        trainer.register_optimizers_hook()

        trainer.invoke_hook(TrainerStages.before_run)

        for _ in range(trainer._epoch, trainer._max_epochs):
            trainer.invoke_hook(TrainerStages.before_train_epoch)
            for _, data_batch in enumerate(train_dataloader):
                for k, v in data_batch.items():
                    data_batch[k] = v.cuda()
                trainer.invoke_hook(TrainerStages.before_train_iter)
                trainer.train_step(trainer.model, data_batch)
                trainer.invoke_hook(TrainerStages.after_train_iter)

                self.assertEqual(trainer.train_outputs['logits'].dtype,
                                 torch.float16)

                # test if `after_train_iter`, whether the model is reset to fp32
                trainer.train_step(trainer.model, data_batch)
                self.assertEqual(trainer.train_outputs['logits'].dtype,
                                 torch.float32)

                self.assertEqual(
                    len(trainer.optimizer.param_groups[0]['params']), 4)
                for i in range(4):
                    self.assertTrue(trainer.optimizer.param_groups[0]['params']
                                    [i].requires_grad)

            trainer.invoke_hook(TrainerStages.after_train_epoch)
            trainer._epoch += 1
        trainer.invoke_hook(TrainerStages.after_run)


if __name__ == '__main__':
    unittest.main()
