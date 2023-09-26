# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class CardDetectionCorrectionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_resnet18_card_correction'
        cache_path = snapshot_download(self.model_id)
        self.test_image = osp.join(cache_path, 'data/demo.jpg')
        self.task = Tasks.card_detection_correction

    def pipeline_inference(self, pipe: Pipeline, input_location: str):
        result = pipe(input_location)
        print('card detection results: ')
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        card_detection_correction = pipeline(
            Tasks.card_detection_correction, model=self.model_id)
        self.pipeline_inference(card_detection_correction, self.test_image)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        card_detection_correction = pipeline(Tasks.card_detection_correction)
        self.pipeline_inference(card_detection_correction, self.test_image)


if __name__ == '__main__':
    unittest.main()
