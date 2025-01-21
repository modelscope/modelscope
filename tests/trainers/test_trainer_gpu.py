# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import os
import shutil
import tempfile
import unittest

import json
import numpy as np
import torch
from packaging import version
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import IterableDataset

from modelscope.metainfo import Metrics, Trainers
from modelscope.metrics.builder import MetricKeys
from modelscope.models.base import TorchModel
from modelscope.trainers import build_trainer
from modelscope.utils.constant import LogKeys, ModeKeys, ModelFile, Tasks
from modelscope.utils.test_utils import (DistributedTestCase,
                                         create_dummy_test_dataset, test_level)


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


class DummyModelForwardInputs(DummyModel):

    def forward(self, inputs):
        feat, labels = inputs['feat'], inputs['labels']
        return super().forward(feat, labels)


def train_func(work_dir,
               dist=False,
               iterable_dataset=False,
               forward_inputs=False,
               **kwargs):
    json_cfg = {
        'task': Tasks.image_classification,
        'model': {},
        'train': {
            'work_dir': work_dir,
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
                'batch_size_per_gpu': 1,
                'workers_per_gpu': 1,
                'shuffle': False
            },
            'metrics': [Metrics.seq_cls_metric]
        }
    }

    config_path = os.path.join(work_dir, ModelFile.CONFIGURATION)
    with open(config_path, 'w') as f:
        json.dump(json_cfg, f)

    if forward_inputs:
        model = DummyModelForwardInputs()
    else:
        model = DummyModel()
    optimmizer = SGD(model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimmizer, 2)
    trainer_name = Trainers.default
    if iterable_dataset:
        train_dataset = DummyIterableDataset()
        eval_dataset = DummyIterableDataset()
    else:
        train_dataset = dummy_dataset_big
        eval_dataset = dummy_dataset_small
    _kwargs = dict(
        cfg_file=config_path,
        model=model,
        data_collator=None,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimmizer, lr_scheduler),
        max_epochs=3,
        device='gpu',
        launcher='pytorch' if dist else None,
        **kwargs)

    trainer = build_trainer(trainer_name, _kwargs)
    trainer.train()


@unittest.skipIf(not torch.cuda.is_available(), 'cuda unittest')
class TrainerTestSingleGpu(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_single_gpu(self):
        train_func(self.tmp_dir)

        results_files = os.listdir(self.tmp_dir)
        json_files = glob.glob(os.path.join(self.tmp_dir, '*.log.json'))
        self.assertEqual(len(json_files), 1)

        with open(json_files[0], 'r', encoding='utf-8') as f:
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
                LogKeys.ITER: 20
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
                LogKeys.ITER: 20
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
                LogKeys.ITER: 20
            }, json.loads(lines[8]))
        self.assertIn(f'{LogKeys.EPOCH}_1.pth', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_2.pth', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_3.pth', results_files)
        for i in [0, 1, 3, 4, 6, 7]:
            self.assertIn(LogKeys.DATA_LOAD_TIME, lines[i])
            self.assertIn(LogKeys.ITER_TIME, lines[i])
        for i in [2, 5, 8]:
            self.assertIn(MetricKeys.ACCURACY, lines[i])


@unittest.skipIf(not torch.cuda.is_available()
                 or torch.cuda.device_count() <= 1, 'distributed unittest')
class TrainerTestMultiGpus(DistributedTestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_multi_gpus(self):
        self.start(train_func, num_gpus=2, work_dir=self.tmp_dir, dist=True)

        results_files = os.listdir(self.tmp_dir)
        json_files = glob.glob(os.path.join(self.tmp_dir, '*.log.json'))
        self.assertEqual(len(json_files), 1)

        with open(json_files[0], 'r', encoding='utf-8') as f:
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
                LogKeys.MODE: ModeKeys.EVAL,
                LogKeys.EPOCH: 1,
                LogKeys.ITER: 10
            }, json.loads(lines[1]))
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.TRAIN,
                LogKeys.EPOCH: 2,
                LogKeys.ITER: 10,
                LogKeys.LR: 0.01
            }, json.loads(lines[2]))
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.EVAL,
                LogKeys.EPOCH: 2,
                LogKeys.ITER: 10
            }, json.loads(lines[3]))
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.TRAIN,
                LogKeys.EPOCH: 3,
                LogKeys.ITER: 10,
                LogKeys.LR: 0.001
            }, json.loads(lines[4]))
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.EVAL,
                LogKeys.EPOCH: 3,
                LogKeys.ITER: 10
            }, json.loads(lines[5]))
        self.assertIn(f'{LogKeys.EPOCH}_1.pth', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_2.pth', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_3.pth', results_files)
        for i in [0, 2, 4]:
            self.assertIn(LogKeys.DATA_LOAD_TIME, lines[i])
            self.assertIn(LogKeys.ITER_TIME, lines[i])
        for i in [1, 3, 5]:
            self.assertIn(MetricKeys.ACCURACY, lines[i])

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_multi_gpus_forward_inputs(self):
        self.start(
            train_func,
            num_gpus=2,
            work_dir=self.tmp_dir,
            dist=True,
            forward_inputs=True)

        results_files = os.listdir(self.tmp_dir)
        json_files = glob.glob(os.path.join(self.tmp_dir, '*.log.json'))
        self.assertEqual(len(json_files), 1)
        self.assertIn(f'{LogKeys.EPOCH}_1.pth', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_2.pth', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_3.pth', results_files)

    # TODO: support iters_per_epoch for dist mode
    @unittest.skipIf(True, 'need to adapt to DistributedSampler')
    def test_multi_gpus_with_iters_per_epoch(self):
        self.start(
            train_func,
            num_gpus=2,
            work_dir=self.tmp_dir,
            dist=True,
            iterable_dataset=True,
            train_iters_per_epoch=20,
            val_iters_per_epoch=10,
        )

        results_files = os.listdir(self.tmp_dir)
        json_files = glob.glob(os.path.join(self.tmp_dir, '*.log.json'))
        self.assertEqual(len(json_files), 1)

        with open(json_files[0], 'r', encoding='utf-8') as f:
            lines = [i.strip() for i in f.readlines()]

        print(results_files, lines)


def train_func_2(work_dir,
                 dist=False,
                 iterable_dataset=False,
                 forward_inputs=False,
                 **kwargs):
    json_cfg = {
        'task': Tasks.image_classification,
        'model': {},
        'train': {
            'work_dir': work_dir,
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
                'batch_size_per_gpu': 1,
                'workers_per_gpu': 1,
                'shuffle': False
            },
            'metrics': [Metrics.seq_cls_metric]
        }
    }

    extra_hooks = [{'type': 'ApexAMPOptimizerHook'}]
    json_cfg['train']['hooks'].extend(extra_hooks)
    config_path = os.path.join(work_dir, ModelFile.CONFIGURATION)
    with open(config_path, 'w') as f:
        json.dump(json_cfg, f)

    if forward_inputs:
        model = DummyModelForwardInputs()
    else:
        model = DummyModel()
    optimmizer = SGD(model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimmizer, 2)
    trainer_name = Trainers.default
    if iterable_dataset:
        train_dataset = DummyIterableDataset()
        eval_dataset = DummyIterableDataset()
    else:
        train_dataset = dummy_dataset_big
        eval_dataset = dummy_dataset_small
    _kwargs = dict(
        cfg_file=config_path,
        model=model,
        data_collator=None,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimmizer, lr_scheduler),
        max_epochs=3,
        device='gpu',
        launcher='pytorch' if dist else None,
        **kwargs)

    trainer = build_trainer(trainer_name, _kwargs)
    trainer.train()
    assert isinstance(trainer.model, DistributedDataParallel)
    assert isinstance(trainer.model.module, DummyModel)
    assert trainer.train_outputs['logits'].dtype == torch.float16


@unittest.skipIf(not torch.cuda.is_available()
                 or torch.cuda.device_count() <= 1
                 or version.parse(torch.__version__) >= version.parse('1.9.0'),
                 'skip on torch 1.9 or above')
class TrainerTestDDPAndApex(DistributedTestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_multi_gpus_apex(self):
        self.start(train_func_2, num_gpus=2, work_dir=self.tmp_dir, dist=True)


def test_func(work_dir,
              dist=False,
              iterable_dataset=False,
              forward_inputs=False,
              **kwargs):
    json_cfg = {
        'task': Tasks.image_classification,
        'model': {},
        'train': {
            'work_dir': work_dir,
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
                'batch_size_per_gpu': 1,
                'workers_per_gpu': 1,
                'shuffle': False
            },
            'metrics': [Metrics.seq_cls_metric]
        }
    }

    config_path = os.path.join(work_dir, ModelFile.CONFIGURATION)
    with open(config_path, 'w') as f:
        json.dump(json_cfg, f)

    if forward_inputs:
        model = DummyModelForwardInputs()
    else:
        model = DummyModel()
    torch.save(model.state_dict(), os.path.join(work_dir, 'pytorch_model.bin'))
    optimmizer = SGD(model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimmizer, 2)
    trainer_name = Trainers.default
    if iterable_dataset:
        train_dataset = DummyIterableDataset()
        eval_dataset = DummyIterableDataset()
    else:
        train_dataset = dummy_dataset_big
        eval_dataset = dummy_dataset_small
    _kwargs = dict(
        cfg_file=config_path,
        model=model,
        data_collator=None,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimmizer, lr_scheduler),
        max_epochs=3,
        device='gpu',
        launcher='pytorch' if dist else None,
        **kwargs)

    trainer = build_trainer(trainer_name, _kwargs)
    trainer.evaluate()
    assert isinstance(trainer.model, DistributedDataParallel)
    assert isinstance(trainer.model.module, DummyModel)
    metric_values = trainer.metric_values
    trainer.evaluate(os.path.join(work_dir, 'pytorch_model.bin'))
    assert isinstance(trainer.model, DistributedDataParallel)
    assert isinstance(trainer.model.module, DummyModel)
    print(metric_values)
    print(trainer.metric_values)
    for key in metric_values:
        assert np.isclose(metric_values[key], trainer.metric_values[key])


@unittest.skipIf(not torch.cuda.is_available()
                 or torch.cuda.device_count() <= 1,
                 'skip on torch 1.9 or above')
class TrainerTestDDPTest(DistributedTestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_multi_gpus_apex_test(self):
        self.start(test_func, num_gpus=2, work_dir=self.tmp_dir, dist=True)


if __name__ == '__main__':
    unittest.main()
