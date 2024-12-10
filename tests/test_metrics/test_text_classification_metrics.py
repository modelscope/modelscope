# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

import numpy as np

from modelscope.metrics.sequence_classification_metric import \
    SequenceClassificationMetric
from modelscope.utils.test_utils import test_level


class TestTextClsMetrics(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_value(self):
        metric = SequenceClassificationMetric()
        outputs = {
            'logits':
            np.array([[2.0, 1.0, 0.5], [1.0, 1.5, 1.0], [2.0, 1.0, 3.0],
                      [2.4, 1.5, 4.0], [2.0, 1.0, 3.0], [2.4, 1.5, 1.7],
                      [2.0, 1.0, 0.5], [2.4, 1.5, 0.5]])
        }
        inputs = {'labels': np.array([0, 1, 2, 2, 0, 1, 2, 2])}
        metric.add(outputs, inputs)
        ret = metric.evaluate()
        self.assertTrue(np.isclose(ret['f1'], 0.5))
        self.assertTrue(np.isclose(ret['accuracy'], 0.5))
        print(ret)


if __name__ == '__main__':
    unittest.main()
