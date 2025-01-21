# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class VideoSummarizationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.video_summarization
        self.model_id = 'damo/cv_googlenet_pgl-video-summarization'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        video_path = 'data/test/videos/video_category_test_video.mp4'
        summarization_pipeline = pipeline(
            Tasks.video_summarization, model=self.model_id)
        result = summarization_pipeline(video_path)

        print(f'video summarization output: \n{result}.')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        video_path = 'data/test/videos/video_category_test_video.mp4'
        summarization_pipeline = pipeline(Tasks.video_summarization)
        result = summarization_pipeline(video_path)

        print(f'video summarization output:\n {result}.')


if __name__ == '__main__':
    unittest.main()
