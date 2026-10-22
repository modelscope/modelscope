# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.metrics.ned_metric import NedMetric
from modelscope.utils.test_utils import test_level


class TestNedMetric(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_an_empty_string_does_not_score_below_zero(self):
        metric = NedMetric()
        metric.preds = ['', 'ab', 'cat', 'yes']
        metric.labels = ['ab', '', 'cut', 'yes']
        score = metric.evaluate()['ned']
        # An empty string against "ab" used to return distance 2, so the
        # score was -1. The normalized distance is 1, so the score is 0.
        # cat/cut is still 2/3 and an exact match is still 1.
        self.assertAlmostEqual(score, (0.0 + 0.0 + (1.0 - 1.0 / 3.0) + 1.0) / 4.0)


if __name__ == '__main__':
    unittest.main()
