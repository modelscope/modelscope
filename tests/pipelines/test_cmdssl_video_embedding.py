# Copyright (c) Alibaba, Inc. and its affiliates.
# !/usr/bin/env python
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class CMDSSLVideoEmbeddingTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        videossl_pipeline = pipeline(
            Tasks.video_embedding, model='damo/cv_r2p1d_video_embedding')
        result = videossl_pipeline(
            'data/test/videos/action_recognition_test_video.mp4')

        print(f'video embedding output: {result}.')


if __name__ == '__main__':
    unittest.main()
