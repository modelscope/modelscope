# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class TinynasObjectDetectionTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run(self):
        tinynas_object_detection = pipeline(
            Tasks.image_object_detection, model='damo/cv_tinynas_detection')
        result = tinynas_object_detection(
            'data/test/images/image_detection.jpg')
        print(result)

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.test_demo()


if __name__ == '__main__':
    unittest.main()
