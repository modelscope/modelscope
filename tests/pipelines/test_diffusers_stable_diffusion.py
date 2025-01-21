# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

import cv2

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class DiffusersStableDiffusionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.text_to_image_synthesis
        self.model_id = 'shadescript/stable-diffusion-2-1-dev'

    test_input = 'a photo of an astronaut riding a horse on mars'

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run(self):
        diffusers_pipeline = pipeline(task=self.task, model=self.model_id)
        output = diffusers_pipeline({
            'text': self.test_input,
            'height': 512,
            'width': 512
        })
        cv2.imwrite('output.png', output['output_imgs'][0])
        print('Image saved to output.png')


if __name__ == '__main__':
    unittest.main()
