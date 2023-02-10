# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class ObjectDetectionTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.image_object_detection
        self.model_id = 'damo/cv_resnet50_object-detection_maskscoring'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_abnormal_object_detection(self):
        input_location = 'data/test/images/image_detection.jpg'
        object_detect = pipeline(self.task, model=self.model_id)
        result = object_detect(input_location)
        print(result)

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
