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
from modelscope.models.base import TorchModel
from modelscope.trainers import build_trainer
from modelscope.utils.constant import ModelFile
from modelscope.utils.registry import default_group
from modelscope.utils.test_utils import create_dummy_test_dataset


def create_dummy_metric():

    @METRICS.register_module(
        group_key=default_group, module_name='DummyMetric', force=True)
    class DummyMetric:

        def add(*args, **kwargs):
            pass

        def evaluate(self):
            return {MetricKeys.ACCURACY: 0.5}


dummy_dataset = create_dummy_test_dataset(
    np.random.random(size=(5, )), np.random.randint(0, 4, (1, )), 20)


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


class EvaluationHookTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        create_dummy_metric()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir)

    def test_evaluation_hook(self):
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
                },
                'lr_scheduler': {
                    'type': 'StepLR',
                    'step_size': 2,
                },
                'hooks': [{
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
            max_epochs=1)

        trainer = build_trainer(trainer_name, kwargs)
        trainer.train()
        self.assertDictEqual(trainer.metric_values, {'accuracy': 0.5})


if __name__ == '__main__':
    unittest.main()
