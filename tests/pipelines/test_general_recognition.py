# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class GeneralRecognitionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.general_recognition
        self.model_id = 'damo/cv_resnest101_general_recognition'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run(self):
        general_recognition = pipeline(
            Tasks.general_recognition,
            model='damo/cv_resnest101_general_recognition')
        result = general_recognition('data/test/images/dogs.jpg')
        print(result)


if __name__ == '__main__':
    unittest.main()
