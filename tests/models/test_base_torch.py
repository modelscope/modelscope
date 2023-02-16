# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import shutil
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.models.base import TorchModel


class TorchBaseTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    def test_custom_model(self):

        class MyTorchModel(TorchModel):

            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, input):
                x = F.relu(self.conv1(input))
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

            def forward(self, input):
                x = F.relu(self.conv1(input))
                return F.relu(self.conv2(x))

            def postprocess(self, x):
                return x + add_bias

        model = MyTorchModel()
        model.train()
        model.eval()
        out = model(torch.rand(1, 1, 10, 10))
        self.assertEqual((1, 20, 2, 2), out.shape)
        self.assertTrue(np.all(out.detach().numpy() > (add_bias - 10)))

    def test_save_pretrained(self):
        model = TorchModel.from_pretrained(
            'damo/nlp_structbert_sentence-similarity_chinese-tiny')
        save_path = os.path.join(self.tmp_dir, 'test_save_pretrained')
        model.save_pretrained(
            save_path, save_checkpoint_names='pytorch_model.bin')
        self.assertTrue(
            os.path.isfile(os.path.join(save_path, 'pytorch_model.bin')))
        self.assertTrue(
            os.path.isfile(os.path.join(save_path, 'configuration.json')))
        self.assertTrue(os.path.isfile(os.path.join(save_path, 'vocab.txt')))


if __name__ == '__main__':
    unittest.main()
