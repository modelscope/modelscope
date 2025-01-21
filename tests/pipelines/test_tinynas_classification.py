# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class TinyNASClassificationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.image_classification
        self.model_id = 'damo/cv_tinynas_classification'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run(self):
        tinynas_classification = pipeline(
            Tasks.image_classification, model='damo/cv_tinynas_classification')
        result = tinynas_classification('data/test/images/image_wolf.jpeg')
        print(result)


if __name__ == '__main__':
    unittest.main()
