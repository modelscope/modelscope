# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class ChineseStableDiffusionTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.text_to_image_synthesis
        self.model_id = 'damo/multi-modal_chinese_stable_diffusion_v1.0'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_default(self):
        pipe = pipeline(task=self.task, model=self.model_id)
        output = pipe('中国山水画')
        output['output_img'][0].save('result.png')
        print('Image saved to result.png')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_dpmsolver(self):
        from diffusers.schedulers import DPMSolverMultistepScheduler
        pipe = pipeline(task=self.task, model=self.model_id)
        pipe.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.pipeline.scheduler.config)
        output = pipe('中国山水画')
        output['output_img'][0].save('result2.png')
        print('Image saved to result2.png')


if __name__ == '__main__':
    unittest.main()
