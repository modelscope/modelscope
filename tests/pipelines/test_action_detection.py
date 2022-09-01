# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.test_utils import test_level


class ActionDetectionTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run(self):
        action_detection_pipline = pipeline(
            Tasks.action_detection,
            model='damo/cv_ResNetC3D_action-detection_detection2d')
        result = action_detection_pipline(
            'data/test/videos/action_detection_test_video.mp4')
        print('action detection results:', result)


if __name__ == '__main__':
    unittest.main()
