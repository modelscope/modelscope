# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
import unittest

import cv2

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ImageColorizationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_unet_image-colorization'
        self.test_image = 'data/test/images/marilyn_monroe_4.jpg'
        self.task = Tasks.image_colorization

    def pipeline_inference(self, pipeline: Pipeline, test_image: str):
        result = pipeline(test_image)
        if result is not None:
            cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
            print(f'Output written to {osp.abspath("result.png")}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        image_colorization = pipeline(
            Tasks.image_colorization, model=self.model_id)

        self.pipeline_inference(image_colorization, self.test_image)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        image_colorization = pipeline(Tasks.image_colorization)
        self.pipeline_inference(image_colorization, self.test_image)


if __name__ == '__main__':
    unittest.main()
