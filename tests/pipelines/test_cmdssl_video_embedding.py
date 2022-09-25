# Copyright (c) Alibaba, Inc. and its affiliates.
# !/usr/bin/env python
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class CMDSSLVideoEmbeddingTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.video_embedding
        self.model_id = 'damo/cv_r2p1d_video_embedding'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        videossl_pipeline = pipeline(task=self.task, model=self.model_id)
        result = videossl_pipeline(
            'data/test/videos/action_recognition_test_video.mp4')

        print(f'video embedding output: {result}.')

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
