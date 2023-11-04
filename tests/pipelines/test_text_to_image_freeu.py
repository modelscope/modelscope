# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

import cv2

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.multi_modal import FreeUTextToImagePipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ImageEditingTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.text_to_image_synthesis
        self.model_id = 'damo/multi-modal_freeu_stable_diffusion'
        prompt = 'a photo of a running corgi'  # prompt
        self.inputs = {'prompt': prompt}
        self.output_image_path = './result.png'
        self.base_model = 'AI-ModelScope/stable-diffusion-v2-1'
        self.freeu_params = {
            'b1': 1.4,
            'b2': 1.6,
            's1': 0.9,
            's2': 0.2
        }  # for SD2.1

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        pipeline = FreeUTextToImagePipeline(cache_path)
        pipeline.group_key = self.task
        synthesized_img = pipeline(
            input=self.inputs)[OutputKeys.OUTPUT_IMGS]  # BGR
        cv2.imwrite(self.output_image_path, synthesized_img)
        print('FreeU pipeline: the synthesized image path is {}'.format(
            self.output_image_path))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.text_to_image_synthesis,
            model=self.model_id,
            base_model=self.base_model,
            freeu_params=self.freeu_params)
        synthesized_img = pipeline_ins(
            self.inputs)[OutputKeys.OUTPUT_IMGS]  # BGR
        cv2.imwrite(self.output_image_path, synthesized_img)
        print('FreeU pipeline: the synthesized image path is {}'.format(
            self.output_image_path))


if __name__ == '__main__':
    unittest.main()
