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


class ImageInpaintingTest(unittest.TestCase):

    def setUp(self) -> None:
        self.input_location = 'data/test/images/image_inpainting/image_inpainting.png'
        self.input_mask_location = 'data/test/images/image_inpainting/image_inpainting_mask.png'
        self.model_id = 'damo/cv_fft_inpainting_lama'

    def save_result(self, result):
        vis_img = result[OutputKeys.OUTPUT_IMG]
        cv2.imwrite('result.png', vis_img)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_inpainting(self):
        inpainting = pipeline(Tasks.image_inpainting, model=self.model_id)
        result = inpainting(self.input_location + '+'
                            + self.input_mask_location)
        if result:
            self.save_result(result)
        else:
            raise ValueError('process error')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    @unittest.skipIf(not torch.cuda.is_available(), 'cuda unittest')
    def test_inpainting_with_refinement(self):
        # if input image is HR, set refine=True is more better
        inpainting = pipeline(
            Tasks.image_inpainting, model=self.model_id, refine=True)
        result = inpainting(self.input_location + '+'
                            + self.input_mask_location)
        if result:
            self.save_result(result)
        else:
            raise ValueError('process error')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_inpainting_with_image(self):
        inpainting = pipeline(Tasks.image_inpainting, model=self.model_id)
        img = Image.open(self.input_location).convert('RGB')
        mask = Image.open(self.input_mask_location).convert('RGB')
        img_new = Image.new('RGB', (img.width + mask.width, img.height))
        img_new.paste(img, (0, 0))
        img_new.paste(mask, (img.width, 0))
        result = inpainting(img_new)
        if result:
            self.save_result(result)
        else:
            raise ValueError('process error')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_inpainting_with_default_task(self):
        inpainting = pipeline(Tasks.image_inpainting)
        result = inpainting(self.input_location + '+'
                            + self.input_mask_location)
        if result:
            self.save_result(result)
        else:
            raise ValueError('process error')


if __name__ == '__main__':
    unittest.main()
