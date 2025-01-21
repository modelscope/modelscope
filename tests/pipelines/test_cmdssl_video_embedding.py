# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
# !/usr/bin/env python
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class CMDSSLVideoEmbeddingTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.video_embedding
        self.model_id = 'damo/cv_r2p1d_video_embedding'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        videossl_pipeline = pipeline(task=self.task, model=self.model_id)
        result = videossl_pipeline(
            'data/test/videos/action_recognition_test_video.mp4')

        print(f'video embedding output: {result}.')


if __name__ == '__main__':
    unittest.main()
