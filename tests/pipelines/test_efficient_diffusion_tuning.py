# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import unittest

from modelscope.models import Model
from modelscope.models.multi_modal import EfficientStableDiffusion
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class EfficientDiffusionTuningTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.efficient_diffusion_tuning

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_efficient_diffusion_tuning_lora_run_pipeline(self):
        model_id = 'damo/multi-modal_efficient-diffusion-tuning-lora'
        inputs = {'prompt': 'pale golden rod circle with old lace background'}
        edt_pipeline = pipeline(self.task, model_id)
        result = edt_pipeline(inputs)
        print(f'Efficient-diffusion-tuning-lora output: {result}.')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_efficient_diffusion_tuning_lora_load_model_from_pretrained(self):
        model_id = 'damo/multi-modal_efficient-diffusion-tuning-lora'
        model = Model.from_pretrained(model_id)
        self.assertTrue(model.__class__ == EfficientStableDiffusion)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_efficient_diffusion_tuning_control_lora_run_pipeline(self):
        model_id = 'damo/multi-modal_efficient-diffusion-tuning-control-lora'
        inputs = {
            'prompt':
            'pale golden rod circle with old lace background',
            'cond':
            'data/test/images/efficient_diffusion_tuning_sd_control_lora_source.png'
        }
        edt_pipeline = pipeline(self.task, model_id)
        result = edt_pipeline(inputs)
        print(f'Efficient-diffusion-tuning-control-lora output: {result}.')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_efficient_diffusion_tuning_control_lora_load_model_from_pretrained(
            self):
        model_id = 'damo/multi-modal_efficient-diffusion-tuning-control-lora'
        model = Model.from_pretrained(model_id)
        self.assertTrue(model.__class__ == EfficientStableDiffusion)


if __name__ == '__main__':
    unittest.main()
