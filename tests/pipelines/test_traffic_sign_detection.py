# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from PIL import Image

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class TrafficSignDetectionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.domain_specific_object_detection
        self.model_id = 'damo/cv_tinynas_object-detection_damoyolo_traffic_sign'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_traffic_sign_detection_damoyolo(self):
        tinynas_object_detection = pipeline(
            Tasks.domain_specific_object_detection,
            model='damo/cv_tinynas_object-detection_damoyolo_traffic_sign')
        result = tinynas_object_detection(
            'data/test/images/image_traffic_sign.jpg')
        assert result and (OutputKeys.SCORES in result) and (
            OutputKeys.LABELS in result) and (OutputKeys.BOXES in result)
        print('results: ', result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_traffic_sign_detection_damoyolo_with_image(self):
        tinynas_object_detection = pipeline(
            Tasks.domain_specific_object_detection,
            model='damo/cv_tinynas_object-detection_damoyolo_traffic_sign')
        img = Image.open('data/test/images/image_traffic_sign.jpg')
        result = tinynas_object_detection(img)
        assert result and (OutputKeys.SCORES in result) and (
            OutputKeys.LABELS in result) and (OutputKeys.BOXES in result)
        print('results: ', result)


if __name__ == '__main__':
    unittest.main()
