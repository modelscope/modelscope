# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class SalientDetectionTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.semantic_segmentation
        self.model_id = 'damo/cv_u2net_salient-detection'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_salient_detection(self):
        input_location = 'data/test/images/image_salient_detection.jpg'
        model_id = 'damo/cv_u2net_salient-detection'
        salient_detect = pipeline(Tasks.semantic_segmentation, model=model_id)
        result = salient_detect(input_location)
        import cv2
        cv2.imwrite(input_location + '_salient.jpg', result[OutputKeys.MASKS])

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
