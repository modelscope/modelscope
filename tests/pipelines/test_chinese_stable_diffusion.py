# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

import cv2

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ChineseStableDiffusionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.text_to_image_synthesis
        self.model_id = 'damo/multi-modal_chinese_stable_diffusion_v1.0'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_default(self):
        pipe = pipeline(task=self.task, model=self.model_id)
        output = pipe({'text': '中国山水画'})
        cv2.imwrite('result.png', output['output_imgs'][0])
        print('Image saved to result.png')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_dpmsolver(self):
        from diffusers.schedulers import DPMSolverMultistepScheduler
        pipe = pipeline(task=self.task, model=self.model_id)
        pipe.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.pipeline.scheduler.config)
        output = pipe({'text': '中国山水画', 'num_inference_steps': 25})
        cv2.imwrite('result2.png', output['output_imgs'][0])
        print('Image saved to result2.png')


if __name__ == '__main__':
    unittest.main()
