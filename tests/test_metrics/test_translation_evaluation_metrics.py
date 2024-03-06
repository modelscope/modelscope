# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.metrics.translation_evaluation_metric import \
    TranslationEvaluationMetric
from modelscope.models.nlp.unite.configuration import InputFormat
from modelscope.utils.test_utils import test_level


class TestTranslationEvaluationMetrics(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_value(self):
        metric = TranslationEvaluationMetric(gap_threshold=25.0)

        outputs = {'score': [0.25, 0.22, 0.30, 0.78, 1.11, 0.95, 1.00, 0.86]}
        inputs = {
            'lp': ['zh-en'] * 8,
            'segment_id': [0, 0, 0, 1, 1, 2, 2, 2],
            'raw_score': [94.0, 60.0, 25.0, 59.5, 90.0, 100.0, 80.0, 60.0],
            'input_format': [InputFormat.SRC_REF] * 8,
        }
        metric.add(outputs, inputs)
        result = metric.evaluate()
        print(result)


if __name__ == '__main__':
    unittest.main()
