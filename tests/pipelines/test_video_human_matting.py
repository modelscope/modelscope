# Copyright (c) Alibaba, Inc. and its affiliates.
import sys
import unittest

from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class VideoHumanMattingTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model = 'damo/cv_effnetv2_video-human-matting'
        self.video_in = 'data/test/videos/video_matting_test.mp4'
        self.video_out = 'matting_out.mp4'
        self.input = {
            'video_input_path': self.video_in,
            'output_path': self.video_out,
        }

    def pipeline_inference(self, pipeline: Pipeline, input):
        result = pipeline(input)
        print('video matting over, results:', result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        video_human_matting = pipeline(
            Tasks.video_human_matting, model=self.model)
        self.pipeline_inference(video_human_matting, self.input)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        video_human_matting = pipeline(Tasks.video_human_matting)
        self.pipeline_inference(video_human_matting, self.input)


if __name__ == '__main__':
    unittest.main()
