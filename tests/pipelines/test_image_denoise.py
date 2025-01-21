# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.cv import ImageDenoisePipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ImageDenoiseTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.image_denoising
        self.model_id = 'damo/cv_nafnet_image-denoise_sidd'

    demo_image_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/noisy-demo-0.png'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        pipeline = ImageDenoisePipeline(cache_path)
        pipeline.group_key = self.task
        denoise_img = pipeline(
            input=self.demo_image_path)[OutputKeys.OUTPUT_IMG]  # BGR
        h, w = denoise_img.shape[:2]
        print('pipeline: the shape of output_img is {}x{}'.format(h, w))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        pipeline_ins = pipeline(task=Tasks.image_denoising, model=model)
        denoise_img = pipeline_ins(
            input=self.demo_image_path)[OutputKeys.OUTPUT_IMG]  # BGR
        h, w = denoise_img.shape[:2]
        print('pipeline: the shape of output_img is {}x{}'.format(h, w))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.image_denoising, model=self.model_id)
        denoise_img = pipeline_ins(
            input=self.demo_image_path)[OutputKeys.OUTPUT_IMG]  # BGR
        h, w = denoise_img.shape[:2]
        print('pipeline: the shape of output_img is {}x{}'.format(h, w))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.image_denoising)
        denoise_img = pipeline_ins(
            input=self.demo_image_path)[OutputKeys.OUTPUT_IMG]  # BGR
        h, w = denoise_img.shape[:2]
        print('pipeline: the shape of output_img is {}x{}'.format(h, w))


if __name__ == '__main__':
    unittest.main()
