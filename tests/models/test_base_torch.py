# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.models.base import TorchModel


class TorchBaseTest(unittest.TestCase):

    def test_custom_model(self):

        class MyTorchModel(TorchModel):

            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))

        model = MyTorchModel()
        model.train()
        model.eval()
        out = model.forward(torch.rand(1, 1, 10, 10))
        self.assertEqual((1, 20, 2, 2), out.shape)

    def test_custom_model_with_postprocess(self):
        add_bias = 200

        class MyTorchModel(TorchModel):

            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))

            def postprocess(self, x):
                return x + add_bias

        model = MyTorchModel()
        model.train()
        model.eval()
        out = model(torch.rand(1, 1, 10, 10))
        self.assertEqual((1, 20, 2, 2), out.shape)
        self.assertTrue(np.all(out.detach().numpy() > (add_bias - 10)))


if __name__ == '__main__':
    unittest.main()
