# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
import shutil
import sys
import tempfile
import unittest
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import PIL

from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class OCRDetectionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_resnet18_ocr-detection-line-level_damo'
        self.test_image = \
            'https://duguang-image-viewer.oss-cn-hangzhou-zmf.aliyuncs.com/' \
            'xixing.tj/165391027548/TB1bKwlHpXXXXc1XXXXXXXXXXXX_%21%210-item_pic.jpg.jpg'

    def pipeline_inference(self, pipeline: Pipeline, input_location: str):
        result = pipeline(input_location)
        print('ocr detection results: ')
        print(result)

    @unittest.skip('deprecated')
    def test_run_modelhub(self):
        ocr_detection = pipeline(Tasks.ocr_detection, model=self.model_id)
        self.pipeline_inference(ocr_detection, self.test_image)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        ocr_detection = pipeline(Tasks.ocr_detection)
        self.pipeline_inference(ocr_detection, self.test_image)


if __name__ == '__main__':
    unittest.main()
