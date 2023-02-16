# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import unittest

from modelscope.models import Model
from modelscope.models.cv.vision_efficient_tuning.vision_efficient_tuning import \
    VisionEfficientTuningModel
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class VisionEfficientTuningPromptTest(unittest.TestCase,
                                      DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.vision_efficient_tuning
        self.model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-prompt'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_pipeline(self):

        petl_pipeline = pipeline(self.task, self.model_id)
        result = petl_pipeline(
            'data/test/images/vision_efficient_tuning_test_1.png')

        print(f'Vision-efficient-tuning-prompt output: {result}.')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_load_model_from_pretrained(self):
        model = Model.from_pretrained(
            'damo/cv_vitb16_classification_vision-efficient-tuning-prompt')
        self.assertTrue(model.__class__ == VisionEfficientTuningModel)


if __name__ == '__main__':
    unittest.main()
