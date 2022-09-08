# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class ActionDetectionTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.action_detection
        self.model_id = 'damo/cv_ResNetC3D_action-detection_detection2d'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run(self):
        action_detection_pipline = pipeline(self.task, model=self.model_id)
        result = action_detection_pipline(
            'data/test/videos/action_detection_test_video.mp4')
        print('action detection results:', result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
