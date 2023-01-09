# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.regress_test_utils import MsRegressTool
from modelscope.utils.test_utils import test_level


class GeneralImageClassificationTest(unittest.TestCase,
                                     DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.image_classification
        self.model_id = 'damo/cv_vit-base_image-classification_Dailylife-labels'
        self.regress_tool = MsRegressTool(baseline=False)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_ImageNet(self):
        general_image_classification = pipeline(
            Tasks.image_classification,
            model='damo/cv_vit-base_image-classification_ImageNet-labels')
        result = general_image_classification('data/test/images/bird.JPEG')
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_Dailylife(self):
        general_image_classification = pipeline(
            Tasks.image_classification,
            model='damo/cv_vit-base_image-classification_Dailylife-labels')
        with self.regress_tool.monitor_module_single_forward(
                general_image_classification.model,
                'vit_base_image_classification'):
            result = general_image_classification('data/test/images/bird.JPEG')
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_nextvit(self):
        nexit_image_classification = pipeline(
            Tasks.image_classification,
            model='damo/cv_nextvit-small_image-classification_Dailylife-labels'
        )
        result = nexit_image_classification('data/test/images/bird.JPEG')
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_convnext(self):
        convnext_image_classification = pipeline(
            Tasks.image_classification,
            model='damo/cv_convnext-base_image-classification_garbage')
        result = convnext_image_classification('data/test/images/banana.jpg')
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_beitv2(self):
        beitv2_image_classification = pipeline(
            Tasks.image_classification,
            model=
            'damo/cv_beitv2-base_image-classification_patch16_224_pt1k_ft22k_in1k'
        )
        result = beitv2_image_classification('data/test/images/bird.JPEG')
        print(result)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_Dailylife_default(self):
        general_image_classification = pipeline(Tasks.image_classification)
        result = general_image_classification('data/test/images/bird.JPEG')
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
