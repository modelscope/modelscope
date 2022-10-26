# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch

from modelscope.outputs import TextClassificationModelOutput
from modelscope.utils.test_utils import test_level


class TestModelOutput(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        super().tearDown()

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_model_outputs(self):
        outputs = TextClassificationModelOutput(logits=torch.Tensor([1]))
        self.assertEqual(outputs['logits'], torch.Tensor([1]))
        self.assertEqual(outputs[0], torch.Tensor([1]))
        self.assertEqual(outputs.logits, torch.Tensor([1]))
        logits, loss = outputs
        self.assertEqual(logits, torch.Tensor([1]))
        self.assertTrue(loss is None)


if __name__ == '__main__':
    unittest.main()
