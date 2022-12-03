# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class LicensePlateDectionTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_resnet18_license-plate-detection_damo'
        self.test_image = 'data/test/images/license_plate_detection.jpg'
        self.task = Tasks.license_plate_detection

    def pipeline_inference(self, pipe: Pipeline, input_location: str):
        result = pipe(input_location)
        print('license plate recognition results: ')
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        license_plate_detection = pipeline(
            Tasks.license_plate_detection, model=self.model_id)
        self.pipeline_inference(license_plate_detection, self.test_image)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        license_plate_detection = pipeline(Tasks.license_plate_detection)
        self.pipeline_inference(license_plate_detection, self.test_image)

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
