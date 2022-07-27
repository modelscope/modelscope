# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from PIL import Image

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.cv import ImageDenoisePipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ImageDenoiseTest(unittest.TestCase):
    model_id = 'damo/cv_nafnet_image-denoise_sidd'
    demo_image_path = 'data/test/images/noisy-demo-1.png'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        pipeline = ImageDenoisePipeline(cache_path)
        denoise_img = pipeline(
            input=self.demo_image_path)[OutputKeys.OUTPUT_IMG]
        denoise_img = Image.fromarray(denoise_img)
        w, h = denoise_img.size
        print('pipeline: the shape of output_img is {}x{}'.format(h, w))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        pipeline_ins = pipeline(task=Tasks.image_denoise, model=model)
        denoise_img = pipeline_ins(
            input=self.demo_image_path)[OutputKeys.OUTPUT_IMG]
        denoise_img = Image.fromarray(denoise_img)
        w, h = denoise_img.size
        print('pipeline: the shape of output_img is {}x{}'.format(h, w))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(task=Tasks.image_denoise, model=self.model_id)
        denoise_img = pipeline_ins(
            input=self.demo_image_path)[OutputKeys.OUTPUT_IMG]
        denoise_img = Image.fromarray(denoise_img)
        w, h = denoise_img.size
        print('pipeline: the shape of output_img is {}x{}'.format(h, w))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.image_denoise)
        denoise_img = pipeline_ins(
            input=self.demo_image_path)[OutputKeys.OUTPUT_IMG]
        denoise_img = Image.fromarray(denoise_img)
        w, h = denoise_img.size
        print('pipeline: the shape of output_img is {}x{}'.format(h, w))


if __name__ == '__main__':
    unittest.main()
