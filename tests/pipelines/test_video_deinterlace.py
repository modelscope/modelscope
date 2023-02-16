# Copyright (c) Alibaba, Inc. and its affiliates.
import sys
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.cv import VideoDeinterlacePipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class VideoDeinterlaceTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.video_deinterlace
        self.model_id = 'damo/cv_unet_video-deinterlace'
        self.test_video = 'data/test/videos/video_deinterlace_test.mp4'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        pipeline = VideoDeinterlacePipeline(cache_path)
        pipeline.group_key = self.task
        out_video_path = pipeline(
            input=self.test_video)[OutputKeys.OUTPUT_VIDEO]
        print('pipeline: the output video path is {}'.format(out_video_path))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_pretrained(self):
        cache_path = Model.from_pretrained(self.model_id)
        pipeline = VideoDeinterlacePipeline(cache_path)
        pipeline.group_key = self.task
        out_video_path = pipeline(
            input=self.test_video)[OutputKeys.OUTPUT_VIDEO]
        print('pipeline: the output video path is {}'.format(out_video_path))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        pipeline_ins = pipeline(
            task=Tasks.video_deinterlace, model=self.model_id)
        out_video_path = pipeline_ins(
            input=self.test_video)[OutputKeys.OUTPUT_VIDEO]
        print('pipeline: the output video path is {}'.format(out_video_path))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.video_deinterlace)
        out_video_path = pipeline_ins(
            input=self.test_video)[OutputKeys.OUTPUT_VIDEO]
        print('pipeline: the output video path is {}'.format(out_video_path))

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
