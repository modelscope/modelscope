# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
import unittest

import cv2

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class ImagePortraitEnhancementTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.image_portrait_enhancement
        self.model_id = 'damo/cv_gpen_image-portrait-enhancement'
        self.test_image = 'data/test/images/Solvay_conference_1927.png'

    def pipeline_inference(self, pipeline: Pipeline, test_image: str):
        result = pipeline(test_image)
        if result is not None:
            cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
            print(f'Output written to {osp.abspath("result.png")}')
        else:
            raise Exception('Testing failed: invalid output')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        face_enhancement = pipeline(
            Tasks.image_portrait_enhancement, model=self.model_id)
        self.pipeline_inference(face_enhancement, self.test_image)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        face_enhancement = pipeline(Tasks.image_portrait_enhancement)
        self.pipeline_inference(face_enhancement, self.test_image)

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
