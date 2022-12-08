# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class TinynasObjectDetectionTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.image_object_detection
        self.model_id = 'damo/cv_tinynas_object-detection_damoyolo'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_airdet(self):
        tinynas_object_detection = pipeline(
            Tasks.image_object_detection, model='damo/cv_tinynas_detection')
        result = tinynas_object_detection(
            'data/test/images/image_detection.jpg')
        print('airdet', result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_damoyolo(self):
        tinynas_object_detection = pipeline(
            Tasks.image_object_detection,
            model='damo/cv_tinynas_object-detection_damoyolo')
        result = tinynas_object_detection(
            'data/test/images/image_detection.jpg')
        print('damoyolo-s', result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_damoyolo_m(self):
        tinynas_object_detection = pipeline(
            Tasks.image_object_detection,
            model='damo/cv_tinynas_object-detection_damoyolo-m')
        result = tinynas_object_detection(
            'data/test/images/image_detection.jpg')
        print('damoyolo-m', result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_damoyolo_t(self):
        tinynas_object_detection = pipeline(
            Tasks.image_object_detection,
            model='damo/cv_tinynas_object-detection_damoyolo-t')
        result = tinynas_object_detection(
            'data/test/images/image_detection.jpg')
        print('damoyolo-t', result)

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_image_object_detection_auto_pipeline(self):
        test_image = 'data/test/images/image_detection.jpg'
        tinynas_object_detection = pipeline(
            Tasks.image_object_detection,
            model='damo/cv_tinynas_object-detection_damoyolo-m')
        result = tinynas_object_detection(test_image)
        tinynas_object_detection.show_result(test_image, result,
                                             'demo_ret.jpg')


if __name__ == '__main__':
    unittest.main()
