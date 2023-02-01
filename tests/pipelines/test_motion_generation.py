# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class MDMMotionGenerationTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.motion_generation
        self.model_id = 'damo/cv_mdm_motion-generation'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run(self):
        motion_generation_pipline = pipeline(self.task, model=self.model_id)
        result = motion_generation_pipline(
            'the person walked forward and is picking up his toolbox')
        print('motion generation data shape:',
              result[OutputKeys.KEYPOINTS].shape)
        print('motion generation video file:', result[OutputKeys.OUTPUT_VIDEO])

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
