# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class ContentCheckTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.image_classification
        self.model_id = 'damo/cv_resnet50_image-classification_cc'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run(self):
        content_check_func = pipeline(self.task, model=self.model_id)
        result = content_check_func('data/test/images/content_check.jpg')
        print(result)

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
