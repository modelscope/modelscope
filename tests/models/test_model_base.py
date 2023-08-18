# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import shutil
import tempfile
import unittest

import torch

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

    def test_from_pretrained_baichuan(self):
        model = Model.from_pretrained(
            'baichuan-inc/Baichuan-13B-Chat',
            revision='v1.0.8',
            torch_dtype=torch.float16,
            device='gpu')
        print(model.__class__.__name__)
        self.assertIsNotNone(model)

    def test_from_pretrained_chatglm2(self):
        model = Model.from_pretrained(
            'ZhipuAI/chatglm2-6b',
            revision='v1.0.7',
            torch_dtype=torch.float16,
            device='gpu')
        print(model.__class__.__name__)
        self.assertIsNotNone(model)

    def test_from_pretrained_ms(self):
        model = Model.from_pretrained(
            'damo/nlp_structbert_sentence-similarity_chinese-tiny',
            device='gpu')
        print(model.__class__.__name__)
        self.assertIsNotNone(model)


if __name__ == '__main__':
    unittest.main()
