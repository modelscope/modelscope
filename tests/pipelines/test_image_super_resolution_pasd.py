# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
import unittest

import cv2

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ImageSuperResolutionPASDTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/PASD_image_super_resolutions'
        self.model_v2_id = 'damo/PASD_v2_image_super_resolutions'
        self.img = 'data/test/images/dogs.jpg'
        self.input = {
            'image': self.img,
            'prompt': '',
            'upscale': 1,
            'fidelity_scale_fg': 1.0,
            'fidelity_scale_bg': 1.0
        }
        self.task = Tasks.image_super_resolution_pasd

    def pipeline_inference(self, pipeline: Pipeline, input: dict):
        result = pipeline(input)
        if result is not None:
            cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
            print(f'Output written to {osp.abspath("result.png")}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        super_resolution = pipeline(
            Tasks.image_super_resolution_pasd, model=self.model_id)

        self.pipeline_inference(super_resolution, self.input)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub_v2(self):
        super_resolution = pipeline(
            Tasks.image_super_resolution_pasd, model=self.model_v2_id)

        self.pipeline_inference(super_resolution, self.input)


if __name__ == '__main__':
    unittest.main()
