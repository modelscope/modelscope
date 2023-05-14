# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
import unittest

import cv2

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class MobileImageSuperResolutionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_ecbsr_image-super-resolution_mobile'
        self.img = 'data/test/images/butterfly_lrx2_y.png'
        self.task = Tasks.image_super_resolution

    def pipeline_inference(self, pipeline: Pipeline, img: str):
        result = pipeline(img)
        if result is not None:
            cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
            print(f'Output written to {osp.abspath("result.png")}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        super_resolution = pipeline(
            Tasks.image_super_resolution, model=self.model_id)

        self.pipeline_inference(super_resolution, self.img)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        super_resolution = pipeline(Tasks.image_super_resolution)
        self.pipeline_inference(super_resolution, self.img)


if __name__ == '__main__':
    unittest.main()
