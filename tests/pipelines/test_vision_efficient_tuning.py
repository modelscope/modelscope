# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import unittest

from modelscope.models import Model
from modelscope.models.cv.vision_efficient_tuning.model import \
    VisionEfficientTuningModel
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class VisionEfficientTuningTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.vision_efficient_tuning

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_vision_efficient_tuning_adapter_run_pipeline(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-adapter'
        img_path = 'data/test/images/vision_efficient_tuning_test_1.png'
        petl_pipeline = pipeline(self.task, model_id)
        result = petl_pipeline(img_path)
        print(f'Vision-efficient-tuning-adapter output: {result}.')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_vision_efficient_tuning_adapter_load_model_from_pretrained(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-adapter'
        model = Model.from_pretrained(model_id)
        self.assertTrue(model.__class__ == VisionEfficientTuningModel)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_vision_efficient_tuning_lora_run_pipeline(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-lora'
        img_path = 'data/test/images/vision_efficient_tuning_test_1.png'
        petl_pipeline = pipeline(self.task, model_id)
        result = petl_pipeline(img_path)
        print(f'Vision-efficient-tuning-lora output: {result}.')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_vision_efficient_tuning_lora_load_model_from_pretrained(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-lora'
        model = Model.from_pretrained(model_id)
        self.assertTrue(model.__class__ == VisionEfficientTuningModel)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_vision_efficient_tuning_prefix_run_pipeline(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-prefix'
        img_path = 'data/test/images/vision_efficient_tuning_test_1.png'
        petl_pipeline = pipeline(self.task, model_id)
        result = petl_pipeline(img_path)
        print(f'Vision-efficient-tuning-prefix output: {result}.')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_vision_efficient_tuning_prefix_load_model_from_pretrained(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-prefix'
        model = Model.from_pretrained(model_id)
        self.assertTrue(model.__class__ == VisionEfficientTuningModel)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_vision_efficient_tuning_prompt_run_pipeline(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-prompt'
        img_path = 'data/test/images/vision_efficient_tuning_test_1.png'
        petl_pipeline = pipeline(self.task, model_id)
        result = petl_pipeline(img_path)
        print(f'Vision-efficient-tuning-prompt output: {result}.')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_vision_efficient_tuning_prompt_load_model_from_pretrained(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-prompt'
        model = Model.from_pretrained(model_id)
        self.assertTrue(model.__class__ == VisionEfficientTuningModel)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_vision_efficient_tuning_bitfit_run_pipeline(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-bitfit'
        img_path = 'data/test/images/vision_efficient_tuning_test_1.png'
        petl_pipeline = pipeline(self.task, model_id)
        result = petl_pipeline(img_path)
        print(f'Vision-efficient-tuning-bitfit output: {result}.')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_vision_efficient_tuning_bitfit_load_model_from_pretrained(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-bitfit'
        model = Model.from_pretrained(model_id)
        self.assertTrue(model.__class__ == VisionEfficientTuningModel)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_vision_efficient_tuning_sidetuning_run_pipeline(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-sidetuning'
        img_path = 'data/test/images/vision_efficient_tuning_test_1.png'
        petl_pipeline = pipeline(self.task, model_id)
        result = petl_pipeline(img_path)
        print(f'Vision-efficient-tuning-sidetuning output: {result}.')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_vision_efficient_tuning_sidetuning_load_model_from_pretrained(
            self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-sidetuning'
        model = Model.from_pretrained(model_id)
        self.assertTrue(model.__class__ == VisionEfficientTuningModel)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_vision_efficient_tuning_utuning_run_pipeline(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-utuning'
        img_path = 'data/test/images/vision_efficient_tuning_test_1.png'
        petl_pipeline = pipeline(self.task, model_id)
        result = petl_pipeline(img_path)
        print(f'Vision-efficient-tuning-utuning output: {result}.')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_vision_efficient_tuning_utuning_load_model_from_pretrained(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-utuning'
        model = Model.from_pretrained(model_id)
        self.assertTrue(model.__class__ == VisionEfficientTuningModel)


if __name__ == '__main__':
    unittest.main()
