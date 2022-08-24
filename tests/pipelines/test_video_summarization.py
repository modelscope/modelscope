# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class VideoSummarizationTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):

        summarization_pipeline = pipeline(
            Tasks.video_summarization,
            model='damo/cv_googlenet_pgl-video-summarization')
        result = summarization_pipeline(
            'data/test/videos/video_category_test_video.mp4')

        print(f'video summarization output: {result}.')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        summarization_pipeline = pipeline(Tasks.video_summarization)
        result = summarization_pipeline(
            'data/test/videos/video_category_test_video.mp4')

        print(f'video summarization output: {result}.')


if __name__ == '__main__':
    unittest.main()
