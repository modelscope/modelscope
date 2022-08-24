# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

import numpy as np

from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class TextToImageSynthesisTest(unittest.TestCase):
    model_id = 'damo/cv_diffusion_text-to-image-synthesis_tiny'
    test_text = {
        'text': '宇航员',
        'generator_ddim_timesteps': 2,
        'upsampler_256_ddim_timesteps': 2,
        'upsampler_1024_ddim_timesteps': 2,
        'debug': True
    }

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        pipe_line_text_to_image_synthesis = pipeline(
            task=Tasks.text_to_image_synthesis, model=model)
        img = pipe_line_text_to_image_synthesis(
            self.test_text)[OutputKeys.OUTPUT_IMG]
        print(np.sum(np.abs(img)))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipe_line_text_to_image_synthesis = pipeline(
            task=Tasks.text_to_image_synthesis, model=self.model_id)
        img = pipe_line_text_to_image_synthesis(
            self.test_text)[OutputKeys.OUTPUT_IMG]
        print(np.sum(np.abs(img)))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipe_line_text_to_image_synthesis = pipeline(
            task=Tasks.text_to_image_synthesis)
        img = pipe_line_text_to_image_synthesis(
            self.test_text)[OutputKeys.OUTPUT_IMG]
        print(np.sum(np.abs(img)))


if __name__ == '__main__':
    unittest.main()
