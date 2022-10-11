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
from torch.optim.lr_scheduler import MultiStepLR

from modelscope.metainfo import Trainers
from modelscope.models.base import Model
from modelscope.trainers import build_trainer
from modelscope.utils.constant import ModelFile, TrainerStages
from modelscope.utils.test_utils import create_dummy_test_dataset

dummy_dataset = create_dummy_test_dataset(
    np.random.random(size=(5, )), np.random.randint(0, 4, (1, )), 10)


class DummyModel(nn.Module, Model):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 10)
        self.bn = nn.BatchNorm1d(10)

    def forward(self, feat, labels):
        x = self.linear(feat)

        x = self.bn(x)
        loss = torch.sum(x)
        return dict(logits=x, loss=loss)


class SparsityHookTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir)

    def test_sparsity_hook(self):
        json_cfg = {
            'task': 'image_classification',
            'train': {
                'work_dir':
                self.tmp_dir,
                'dataloader': {
                    'batch_size_per_gpu': 2,
                    'workers_per_gpu': 1
                },
                'hooks': [{
                    'type': 'SparsityHook',
                    'pruning_method': 'pst',
                    'config': {
                        'weight_rank': 1,
                        'mask_rank': 1,
                        'final_sparsity': 0.9,
                        'frequency': 1,
                    },
                }],
            },
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
            device='cpu',
        )

        trainer = build_trainer(trainer_name, kwargs)
        train_dataloader = trainer._build_dataloader_with_dataset(
            trainer.train_dataset, **trainer.cfg.train.get('dataloader', {}))
        trainer.register_optimizers_hook()
        trainer.register_hook_from_cfg(trainer.cfg.train.hooks)
        trainer.train_dataloader = train_dataloader
        trainer.data_loader = train_dataloader
        trainer.invoke_hook(TrainerStages.before_run)
        for i in range(trainer._epoch, trainer._max_epochs):
            trainer.invoke_hook(TrainerStages.before_train_epoch)
            for _, data_batch in enumerate(train_dataloader):
                trainer.invoke_hook(TrainerStages.before_train_iter)
                trainer.train_step(trainer.model, data_batch)
                trainer.invoke_hook(TrainerStages.after_train_iter)
            trainer.invoke_hook(TrainerStages.after_train_epoch)
        trainer.invoke_hook(TrainerStages.after_run)

        self.assertEqual(
            torch.mean(1.0 * (trainer.model.linear.weight == 0)), 0.9)


if __name__ == '__main__':
    unittest.main()
