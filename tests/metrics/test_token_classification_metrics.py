# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

import numpy as np

from modelscope.metrics.token_classification_metric import \
    TokenClassificationMetric
from modelscope.utils.test_utils import test_level


class TestTokenClsMetrics(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_value(self):
        metric = TokenClassificationMetric()

        class Trainer:
            pass

        metric.trainer = Trainer()
        metric.trainer.label2id = {
            'B-obj': 0,
            'I-obj': 1,
            'O': 2,
        }

        outputs = {
            'logits':
            np.array([[[2.0, 1.0, 0.5], [1.0, 1.5, 1.0], [2.0, 1.0, 3.0],
                       [2.4, 1.5, 4.0], [2.0, 1.0, 3.0], [2.4, 1.5, 1.7],
                       [2.0, 1.0, 0.5], [2.4, 1.5, 0.5]]])
        }
        inputs = {'labels': np.array([[0, 1, 2, 2, 0, 1, 2, 2]])}
        metric.add(outputs, inputs)
        ret = metric.evaluate()
        self.assertTrue(np.isclose(ret['precision'], 0.25))
        self.assertTrue(np.isclose(ret['recall'], 0.5))
        self.assertTrue(np.isclose(ret['accuracy'], 0.5))
        print(ret)


if __name__ == '__main__':
    unittest.main()
