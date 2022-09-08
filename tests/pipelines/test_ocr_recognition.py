# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import PIL

from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class OCRRecognitionTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_convnextTiny_ocr-recognition-general_damo'
        self.test_image = 'data/test/images/ocr_recognition.jpg'
        self.task = Tasks.ocr_recognition

    def pipeline_inference(self, pipeline: Pipeline, input_location: str):
        result = pipeline(input_location)
        print('ocr recognition results: ', result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        ocr_recognition = pipeline(Tasks.ocr_recognition, model=self.model_id)
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub_PILinput(self):
        ocr_recognition = pipeline(Tasks.ocr_recognition, model=self.model_id)
        imagePIL = PIL.Image.open(self.test_image)
        self.pipeline_inference(ocr_recognition, imagePIL)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        ocr_recognition = pipeline(Tasks.ocr_recognition)
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
