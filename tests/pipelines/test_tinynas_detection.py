# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from PIL import Image

from modelscope.outputs import OutputKeys
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

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_human_detection_damoyolo(self):
        tinynas_object_detection = pipeline(
            Tasks.domain_specific_object_detection,
            model='damo/cv_tinynas_human-detection_damoyolo')
        result = tinynas_object_detection(
            'data/test/images/image_detection.jpg')
        assert result and (OutputKeys.SCORES in result) and (
            OutputKeys.LABELS in result) and (OutputKeys.BOXES in result)
        print('results: ', result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_human_detection_damoyolo_with_image(self):
        tinynas_object_detection = pipeline(
            Tasks.domain_specific_object_detection,
            model='damo/cv_tinynas_human-detection_damoyolo')
        img = Image.open('data/test/images/image_detection.jpg')
        result = tinynas_object_detection(img)
        assert result and (OutputKeys.SCORES in result) and (
            OutputKeys.LABELS in result) and (OutputKeys.BOXES in result)
        print('results: ', result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_facemask_detection_damoyolo(self):
        tinynas_object_detection = pipeline(
            Tasks.domain_specific_object_detection,
            model='damo/cv_tinynas_object-detection_damoyolo_facemask')
        result = tinynas_object_detection(
            'data/test/images/image_detection.jpg')
        assert result and (OutputKeys.SCORES in result) and (
            OutputKeys.LABELS in result) and (OutputKeys.BOXES in result)
        print('results: ', result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_facemask_detection_damoyolo_with_image(self):
        tinynas_object_detection = pipeline(
            Tasks.domain_specific_object_detection,
            model='damo/cv_tinynas_object-detection_damoyolo_facemask')
        img = Image.open('data/test/images/image_detection.jpg')
        result = tinynas_object_detection(img)
        assert result and (OutputKeys.SCORES in result) and (
            OutputKeys.LABELS in result) and (OutputKeys.BOXES in result)
        print('results: ', result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_safetyhat_detection_damoyolo(self):
        tinynas_object_detection = pipeline(
            Tasks.domain_specific_object_detection,
            model='damo/cv_tinynas_object-detection_damoyolo_safety-helmet')
        result = tinynas_object_detection(
            'data/test/images/image_safetyhat.jpg')
        assert result and (OutputKeys.SCORES in result) and (
            OutputKeys.LABELS in result) and (OutputKeys.BOXES in result)
        print('results: ', result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_safetyhat_detection_damoyolo_with_image(self):
        tinynas_object_detection = pipeline(
            Tasks.domain_specific_object_detection,
            model='damo/cv_tinynas_object-detection_damoyolo_safety-helmet')
        img = Image.open('data/test/images/image_safetyhat.jpg')
        result = tinynas_object_detection(img)
        assert result and (OutputKeys.SCORES in result) and (
            OutputKeys.LABELS in result) and (OutputKeys.BOXES in result)
        print('results: ', result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_cigarette_detection_damoyolo(self):
        tinynas_object_detection = pipeline(
            Tasks.domain_specific_object_detection,
            model='damo/cv_tinynas_object-detection_damoyolo_cigarette')
        result = tinynas_object_detection('data/test/images/image_smoke.jpg')
        assert result and (OutputKeys.SCORES in result) and (
            OutputKeys.LABELS in result) and (OutputKeys.BOXES in result)
        print('results: ', result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_cigarette_detection_damoyolo_with_image(self):
        tinynas_object_detection = pipeline(
            Tasks.domain_specific_object_detection,
            model='damo/cv_tinynas_object-detection_damoyolo_cigarette')
        img = Image.open('data/test/images/image_smoke.jpg')
        result = tinynas_object_detection(img)
        assert result and (OutputKeys.SCORES in result) and (
            OutputKeys.LABELS in result) and (OutputKeys.BOXES in result)
        print('results: ', result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_phone_detection_damoyolo(self):
        tinynas_object_detection = pipeline(
            Tasks.domain_specific_object_detection,
            model='damo/cv_tinynas_object-detection_damoyolo_phone')
        result = tinynas_object_detection('data/test/images/image_phone.jpg')
        assert result and (OutputKeys.SCORES in result) and (
            OutputKeys.LABELS in result) and (OutputKeys.BOXES in result)
        print('results: ', result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_phone_detection_damoyolo_with_image(self):
        tinynas_object_detection = pipeline(
            Tasks.domain_specific_object_detection,
            model='damo/cv_tinynas_object-detection_damoyolo_phone')
        img = Image.open('data/test/images/image_phone.jpg')
        result = tinynas_object_detection(img)
        assert result and (OutputKeys.SCORES in result) and (
            OutputKeys.LABELS in result) and (OutputKeys.BOXES in result)
        print('results: ', result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_head_detection_damoyolo(self):
        tinynas_object_detection = pipeline(
            Tasks.domain_specific_object_detection,
            model='damo/cv_tinynas_head-detection_damoyolo')
        result = tinynas_object_detection(
            'data/test/images/image_detection.jpg')
        assert result and (OutputKeys.SCORES in result) and (
            OutputKeys.LABELS in result) and (OutputKeys.BOXES in result)
        print('results: ', result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_head_detection_damoyolo_with_image(self):
        tinynas_object_detection = pipeline(
            Tasks.domain_specific_object_detection,
            model='damo/cv_tinynas_head-detection_damoyolo')
        img = Image.open('data/test/images/image_detection.jpg')
        result = tinynas_object_detection(img)
        assert result and (OutputKeys.SCORES in result) and (
            OutputKeys.LABELS in result) and (OutputKeys.BOXES in result)
        print('results: ', result)


if __name__ == '__main__':
    unittest.main()
