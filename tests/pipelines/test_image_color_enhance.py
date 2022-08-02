# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
import unittest

import cv2

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ImageColorEnhanceTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_csrnet_image-color-enhance-models'

    def pipeline_inference(self, pipeline: Pipeline, input_location: str):
        result = pipeline(input_location)
        if result is not None:
            cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG][:, :,
                                                                    [2, 1, 0]])
            print(f'Output written to {osp.abspath("result.png")}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        img_color_enhance = pipeline(
            Tasks.image_color_enhancement, model=self.model_id)
        self.pipeline_inference(img_color_enhance,
                                'data/test/images/image_color_enhance.png')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        img_color_enhance = pipeline(Tasks.image_color_enhancement)
        self.pipeline_inference(img_color_enhance,
                                'data/test/images/image_color_enhance.png')


if __name__ == '__main__':
    unittest.main()
