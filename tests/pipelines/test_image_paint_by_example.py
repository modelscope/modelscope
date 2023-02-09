# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import cv2
import torch
from PIL import Image

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()


class ImagePaintbyexampleTest(unittest.TestCase):

    def setUp(self) -> None:
        self.input_location = 'data/test/images/image_paint_by_example/image/example_1.png'
        self.input_mask_location = 'data/test/images/image_paint_by_example/mask/example_1.png'
        self.reference_location = 'data/test/images/image_paint_by_example/reference/example_1.jpg'
        self.model_id = 'damo/cv_stable-diffusion_paint-by-example'
        self.input = {
            'img': self.input_location,
            'mask': self.input_mask_location,
            'reference': self.reference_location
        }

    def save_result(self, result):
        vis_img = result[OutputKeys.OUTPUT_IMG]
        vis_img.save('result.png')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_paintbyexample(self):
        paintbyexample = pipeline(
            Tasks.image_paintbyexample, model=self.model_id)
        result = paintbyexample(self.input)
        if result:
            self.save_result(result)
        else:
            raise ValueError('process error')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_paintbyexample_with_image(self):
        paintbyexample = pipeline(
            Tasks.image_paintbyexample, model=self.model_id)
        img = Image.open(self.input_location)
        mask = Image.open(self.input_mask_location)
        reference = Image.open(self.reference_location)
        result = paintbyexample({
            'img': img,
            'mask': mask,
            'reference': reference
        })
        if result:
            self.save_result(result)
        else:
            raise ValueError('process error')


if __name__ == '__main__':
    unittest.main()
