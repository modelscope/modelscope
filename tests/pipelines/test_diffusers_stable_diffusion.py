# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class DiffusersStableDiffusionTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.diffusers_stable_diffusion
        self.model_id = 'shadescript/stable-diffusion-2-1-dev'

    test_input = 'a photo of an astronaut riding a horse on mars'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run(self):
        diffusers_pipeline = pipeline(task=self.task, model=self.model_id)
        output = diffusers_pipeline(self.test_input, height=512, width=512)
        output.images[0].save('/tmp/output.png')
        print('Image saved to /tmp/output.png')


if __name__ == '__main__':
    unittest.main()
