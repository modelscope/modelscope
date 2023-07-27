# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import shutil
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.models.base import Model


class BaseTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    def test_from_pretrained(self):
        model = Model.from_pretrained(
            'baichuan-inc/baichuan-7B', revision='v1.0.5')
        self.assertIsNotNone(model)

    def test_from_pretrained_hf(self):
        model = Model.from_pretrained(
            'damo/nlp_structbert_sentence-similarity_chinese-tiny',
            use_hf=True)
        self.assertIsNotNone(model)


if __name__ == '__main__':
    unittest.main()
