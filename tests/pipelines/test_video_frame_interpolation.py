# Copyright (c) Alibaba, Inc. and its affiliates.
import sys
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.cv import VideoFrameInterpolationPipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class VideoFrameInterpolationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.video_frame_interpolation
        self.model_id = 'damo/cv_raft_video-frame-interpolation'
        self.model_practical_id = 'damo/cv_raft_video-frame-interpolation_practical'
        self.test_video = 'data/test/videos/video_frame_interpolation_test.mp4'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        pipeline = VideoFrameInterpolationPipeline(cache_path)
        pipeline.group_key = self.task
        out_video_path = pipeline(
            input=self.test_video)[OutputKeys.OUTPUT_VIDEO]
        print('pipeline: the output video path is {}'.format(out_video_path))

        cache_path = snapshot_download(self.model_practical_id)
        pipeline = VideoFrameInterpolationPipeline(cache_path)
        pipeline.group_key = self.task
        out_video_path = pipeline(
            input={'video': self.test_video})[OutputKeys.OUTPUT_VIDEO]
        print('pipeline: the output video path is {}'.format(out_video_path))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        pipeline_ins = pipeline(
            task=Tasks.video_frame_interpolation, model=self.model_id)
        out_video_path = pipeline_ins(
            input=self.test_video)[OutputKeys.OUTPUT_VIDEO]
        print('pipeline: the output video path is {}'.format(out_video_path))

        pipeline_ins = pipeline(
            task=Tasks.video_frame_interpolation,
            model=self.model_practical_id)
        out_video_path = pipeline_ins(
            input={'video': self.test_video})[OutputKeys.OUTPUT_VIDEO]
        print('pipeline: the output video path is {}'.format(out_video_path))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.video_frame_interpolation)
        out_video_path = pipeline_ins(
            input=self.test_video)[OutputKeys.OUTPUT_VIDEO]
        print('pipeline: the output video path is {}'.format(out_video_path))


if __name__ == '__main__':
    unittest.main()
