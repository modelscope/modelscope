# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ContentCheckTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.image_classification
        self.model_id = 'damo/cv_resnet50_image-classification_cc'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run(self):
        content_check_func = pipeline(self.task, model=self.model_id)
        result = content_check_func('data/test/images/content_check.jpg')
        print(result)


if __name__ == '__main__':
    unittest.main()
