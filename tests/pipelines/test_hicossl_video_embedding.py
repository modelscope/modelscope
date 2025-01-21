# Copyright (c) Alibaba, Inc. and its affiliates.
# !/usr/bin/env python
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class HICOSSLVideoEmbeddingTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.video_embedding
        self.model_id = 'damo/cv_s3dg_video-embedding'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        videossl_pipeline = pipeline(
            Tasks.video_embedding, model=self.model_id)
        result = videossl_pipeline(
            'data/test/videos/action_recognition_test_video.mp4')

        print(f'video embedding output: {result}.')


if __name__ == '__main__':
    unittest.main()
