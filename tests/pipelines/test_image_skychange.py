# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
import unittest

import cv2
import torch

import modelscope
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

print(modelscope.version.__release_datetime__)


class ImageSkychangeTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model = 'damo/cv_hrnetocr_skychange'
        self.sky_image = 'data/test/images/sky_image.jpg'
        self.scene_image = 'data/test/images/scene_image.jpg'
        self.input = {
            'sky_image': self.sky_image,
            'scene_image': self.scene_image,
        }

    def pipeline_inference(self, pipeline: Pipeline, input: str):
        result = pipeline(input)
        if result is not None:
            cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
            print(f'Output written to {osp.abspath("result.png")}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        image_skychange = pipeline(Tasks.image_skychange, model=self.model)
        self.pipeline_inference(image_skychange, self.input)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        image_skychange = pipeline(Tasks.image_skychange)
        self.pipeline_inference(image_skychange, self.input)


if __name__ == '__main__':
    unittest.main()
