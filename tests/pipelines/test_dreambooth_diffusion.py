# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import unittest

from modelscope.models import Model
from modelscope.models.multi_modal import EfficientStableDiffusion
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class DreamboothDiffusionTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.text_to_image_synthesis

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_dreambooth_diffusion_pipeline(self):
        model_id = 'dreambooth_diffusion_model'
        inputs = {'prompt': 'a dog with old lace background'}
        edt_pipeline = pipeline(self.task, model_id)
        result = edt_pipeline(inputs)
        print(f'Dreambooth-diffusion output: {result}.')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_dreambooth_diffusion_load_model_from_pretrained(self):
        model_id = 'dreambooth_diffusion_model'
        model = Model.from_pretrained(model_id)
        self.assertTrue(model.__class__ == EfficientStableDiffusion)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_dreambooth_diffusion_lora_demo_compatibility(self):
        self.model_id = 'dreambooth_diffusion_model'
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
