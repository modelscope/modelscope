# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class ActionRecognitionTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.action_recognition
        self.model_id = 'damo/cv_TAdaConv_action-recognition'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        recognition_pipeline = pipeline(self.task, self.model_id)
        result = recognition_pipeline(
            'data/test/videos/action_recognition_test_video.mp4')

        print(f'recognition output: {result}.')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        recognition_pipeline = pipeline(self.task)
        result = recognition_pipeline(
            'data/test/videos/action_recognition_test_video.mp4')

        print(f'recognition output: {result}.')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
