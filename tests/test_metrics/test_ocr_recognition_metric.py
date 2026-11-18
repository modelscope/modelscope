# Copyright (c) Alibaba, Inc. and its affiliates.

import pickle
import unittest

import torch

from modelscope.metrics.ocr_recognition_metric import OCRRecognitionMetric
from modelscope.trainers.utils.inference import merge_metrics
from modelscope.utils.test_utils import test_level


def _build_metric(preds, labels, loss):
    metric = OCRRecognitionMetric()
    metric.add({
        'preds': preds,
        'loss': torch.tensor(loss)
    }, {'labels': labels})
    return metric


class TestOCRRecognitionMetric(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_pickle_keeps_state(self):
        metric = _build_metric(['abc', 'xyz'], ['abc', 'xyy'], 0.5)
        restored = pickle.loads(pickle.dumps(metric))
        self.assertEqual(restored.evaluate(), metric.evaluate())

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_merge_parts_from_two_ranks(self):
        # Distributed evaluation pickles each rank's metric objects and
        # merges them on rank 0 with merge_metrics.
        rank0 = _build_metric(['abc', 'xyz'], ['abc', 'xyy'], 0.5)
        rank1 = _build_metric(['hello', 'ok'], ['hello', 'no'], 1.5)
        parts = [
            pickle.loads(pickle.dumps([rank0])),
            pickle.loads(pickle.dumps([rank1]))
        ]
        result = merge_metrics(parts)[0].evaluate()

        # 2 of 4 samples match exactly; 3 edits over 13 target chars.
        self.assertEqual(result['accuracy'], 0.5)
        self.assertAlmostEqual(result['AR'], 1 - 3 / 13)
        self.assertAlmostEqual(result['avg_loss'], 1.0)


if __name__ == '__main__':
    unittest.main()
