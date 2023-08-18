# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import tempfile
import unittest

import cv2

from modelscope.models import Model
from modelscope.models.multi_modal import EfficientStableDiffusion
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class EfficientDiffusionTuningTestSwift(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.efficient_diffusion_tuning

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_efficient_diffusion_tuning_swift_lora_run_pipeline(self):
        model_id = 'damo/multi-modal_efficient-diffusion-tuning-swift-lora'
        model_revision = 'v1.0.2'
        inputs = {
            'prompt':
            'a street scene with a cafe and a restaurant sign in anime style'
        }
        sd_tuner_pipeline = pipeline(self.task, model_id, model_revision=model_revision)
        result = sd_tuner_pipeline(inputs, generator_seed=0)
        output_image_path = tempfile.NamedTemporaryFile(suffix='.png').name
        cv2.imwrite(output_image_path, result['output_imgs'][0])
        print(
            f'Efficient-diffusion-tuning-swift-lora output: {output_image_path}'
        )

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_efficient_diffusion_tuning_swift_lora_load_model_from_pretrained(
            self):
        model_id = 'damo/multi-modal_efficient-diffusion-tuning-swift-lora'
        model_revision = 'v1.0.2'
        model = Model.from_pretrained(model_id, model_revision=model_revision)
        self.assertTrue(model.__class__ == EfficientStableDiffusion)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_efficient_diffusion_tuning_swift_adapter_run_pipeline(self):
        model_id = 'damo/multi-modal_efficient-diffusion-tuning-swift-adapter'
        model_revision = 'v1.0.2'
        inputs = {
            'prompt':
            'a street scene with a cafe and a restaurant sign in anime style'
        }
        sd_tuner_pipeline = pipeline(self.task, model_id, model_revision=model_revision)
        result = sd_tuner_pipeline(inputs, generator_seed=0)
        output_image_path = tempfile.NamedTemporaryFile(suffix='.png').name
        cv2.imwrite(output_image_path, result['output_imgs'][0])
        print(
            f'Efficient-diffusion-tuning-swift-adapter output: {output_image_path}'
        )

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_efficient_diffusion_tuning_swift_adapter_load_model_from_pretrained(
            self):
        model_id = 'damo/multi-modal_efficient-diffusion-tuning-swift-adapter'
        model_revision = 'v1.0.2'
        model = Model.from_pretrained(model_id, model_revision=model_revision)
        self.assertTrue(model.__class__ == EfficientStableDiffusion)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_efficient_diffusion_tuning_swift_prompt_run_pipeline(self):
        model_id = 'damo/multi-modal_efficient-diffusion-tuning-swift-prompt'
        model_revision = 'v1.0.2'
        inputs = {
            'prompt':
            'a street scene with a cafe and a restaurant sign in anime style'
        }
        sd_tuner_pipeline = pipeline(self.task, model_id, model_revision=model_revision)
        result = sd_tuner_pipeline(inputs, generator_seed=0)
        output_image_path = tempfile.NamedTemporaryFile(suffix='.png').name
        cv2.imwrite(output_image_path, result['output_imgs'][0])
        print(
            f'Efficient-diffusion-tuning-swift-prompt output: {output_image_path}'
        )

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_efficient_diffusion_tuning_swift_prompt_load_model_from_pretrained(
            self):
        model_id = 'damo/multi-modal_efficient-diffusion-tuning-swift-prompt'
        model_revision = 'v1.0.2'
        model = Model.from_pretrained(model_id, model_revision=model_revision)
        self.assertTrue(model.__class__ == EfficientStableDiffusion)


if __name__ == '__main__':
    unittest.main()
