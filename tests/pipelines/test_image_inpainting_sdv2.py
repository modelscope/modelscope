# Copyright (c) Alibaba, Inc. and its affiliates.
import tempfile
import unittest

import cv2

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.cv import ImageInpaintingSDV2Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ImageInpaintingSDV2Test(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.image_inpainting
        self.model_id = 'damo/cv_stable-diffusion-v2_image-inpainting_base'
        self.input_location = 'data/test/images/image_inpainting/image_inpainting_1.png'
        self.input_mask_location = 'data/test/images/image_inpainting/image_inpainting_mask_1.png'
        self.prompt = 'background'

        self.input = {
            'image': self.input_location,
            'mask': self.input_mask_location,
            'prompt': self.prompt
        }

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        output_image_path = tempfile.NamedTemporaryFile(suffix='.png').name
        cache_path = snapshot_download(self.model_id)
        pipeline = ImageInpaintingSDV2Pipeline(cache_path)
        pipeline.group_key = self.task
        output = pipeline(input=self.input)[OutputKeys.OUTPUT_IMG]
        cv2.imwrite(output_image_path, output)
        print(
            'pipeline: the output image path is {}'.format(output_image_path))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        output_image_path = tempfile.NamedTemporaryFile(suffix='.png').name
        pipeline_ins = pipeline(
            task=Tasks.image_inpainting, model=self.model_id)
        output = pipeline_ins(input=self.input)[OutputKeys.OUTPUT_IMG]
        cv2.imwrite(output_image_path, output)
        print(
            'pipeline: the output image path is {}'.format(output_image_path))


if __name__ == '__main__':
    unittest.main()
