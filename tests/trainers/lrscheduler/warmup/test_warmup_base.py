# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR


class WarmupTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_constant_warmup(self):
        from modelscope.trainers.lrscheduler.warmup import ConstantWarmup

        net = nn.Linear(2, 2)
        base_lr = 0.02
        warmup_iters = 3
        warmup_ratio = 0.2
        optimizer = torch.optim.SGD(net.parameters(), lr=base_lr, momentum=0.9)
        lr_scheduler = MultiStepLR(optimizer, milestones=[7, 9])
        lr_scheduler_with_warmup = ConstantWarmup(
            lr_scheduler, warmup_iters=warmup_iters, warmup_ratio=warmup_ratio)

        res = []
        for _ in range(10):
            lr_scheduler_with_warmup.step()
            for _, group in enumerate(optimizer.param_groups):
                res.append(group['lr'])

        base_lrs = [0.02, 0.02, 0.02, 0.002, 0.002, 0.0002, 0.0002]
        self.assertListEqual(res, [0.004, 0.004, 0.02] + base_lrs)

    def test_linear_warmup(self):
        from modelscope.trainers.lrscheduler.warmup import LinearWarmup

        net = nn.Linear(2, 2)
        base_lr = 0.02
        warmup_iters = 3
        warmup_ratio = 0.1
        optimizer = torch.optim.SGD(net.parameters(), lr=base_lr, momentum=0.9)
        lr_scheduler = MultiStepLR(optimizer, milestones=[7, 9])
        lr_scheduler_with_warmup = LinearWarmup(
            lr_scheduler, warmup_iters=warmup_iters, warmup_ratio=warmup_ratio)

        res = []
        for _ in range(10):
            lr_scheduler_with_warmup.step()
            for _, group in enumerate(optimizer.param_groups):
                res.append(round(group['lr'], 5))

        base_lrs = [0.02, 0.02, 0.02, 0.002, 0.002, 0.0002, 0.0002]
        self.assertListEqual(res, [0.0080, 0.0140, 0.02] + base_lrs)

    def test_exp_warmup(self):
        from modelscope.trainers.lrscheduler.warmup import ExponentialWarmup

        net = nn.Linear(2, 2)
        base_lr = 0.02
        warmup_iters = 3
        warmup_ratio = 0.1
        optimizer = torch.optim.SGD(net.parameters(), lr=base_lr, momentum=0.9)
        lr_scheduler = MultiStepLR(optimizer, milestones=[7, 9])
        lr_scheduler_with_warmup = ExponentialWarmup(
            lr_scheduler, warmup_iters=warmup_iters, warmup_ratio=warmup_ratio)

        res = []
        for _ in range(10):
            lr_scheduler_with_warmup.step()
            for _, group in enumerate(optimizer.param_groups):
                res.append(round(group['lr'], 5))

        base_lrs = [0.02, 0.02, 0.02, 0.002, 0.002, 0.0002, 0.0002]
        self.assertListEqual(res, [0.00431, 0.00928, 0.02] + base_lrs)


if __name__ == '__main__':
    unittest.main()
