# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader

from modelscope.metrics.builder import MetricKeys
from modelscope.metrics.sequence_classification_metric import \
    SequenceClassificationMetric
from modelscope.models.base import Model
from modelscope.trainers.utils.inference import multi_gpu_test, single_gpu_test
from modelscope.utils.test_utils import (DistributedTestCase,
                                         create_dummy_test_dataset, test_level)
from modelscope.utils.torch_utils import get_dist_info, init_dist

dummy_dataset = create_dummy_test_dataset(
    torch.rand((5, )), torch.randint(0, 4, (1, )), 20)


class DummyModel(nn.Module, Model):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 4)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, feat, labels):
        x = self.linear(feat)

        x = self.bn(x)
        loss = torch.sum(x)
        return dict(logits=x, loss=loss)


def test_func(dist=False):
    dummy_model = DummyModel()
    dataset = dummy_dataset.to_torch_dataset()

    dummy_loader = DataLoader(
        dataset,
        batch_size=2,
    )

    metric_class = SequenceClassificationMetric()

    if dist:
        init_dist(launcher='pytorch')

    rank, world_size = get_dist_info()
    device = torch.device(f'cuda:{rank}')
    dummy_model.cuda()

    if world_size > 1:
        from torch.nn.parallel.distributed import DistributedDataParallel
        dummy_model = DistributedDataParallel(
            dummy_model, device_ids=[torch.cuda.current_device()])
        test_func = multi_gpu_test
    else:
        test_func = single_gpu_test

    metric_results = test_func(
        dummy_model,
        dummy_loader,
        device=device,
        metric_classes=[metric_class])

    return metric_results


@unittest.skipIf(not torch.cuda.is_available(), 'cuda unittest')
class SingleGpuTestTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_single_gpu_test(self):
        metric_results = test_func()
        self.assertIn(MetricKeys.ACCURACY, metric_results)


@unittest.skipIf(not torch.cuda.is_available()
                 or torch.cuda.device_count() <= 1, 'distributed unittest')
class MultiGpuTestTest(DistributedTestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_multi_gpu_test(self):
        self.start(
            test_func,
            num_gpus=2,
            assert_callback=lambda x: self.assertIn(MetricKeys.ACCURACY, x),
            dist=True)


if __name__ == '__main__':
    unittest.main()
