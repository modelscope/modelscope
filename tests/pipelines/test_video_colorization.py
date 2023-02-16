# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
import unittest

import cv2

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.cv import VideoColorizationPipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class VideoColorizationTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.video_colorization
        self.model_id = 'damo/cv_unet_video-colorization'
        self.test_video = 'data/test/videos/video_frame_interpolation_test.mp4'

    def pipeline_inference(self, pipeline: Pipeline, test_video: str):
        result = pipeline(test_video)[OutputKeys.OUTPUT_VIDEO]
        if result is not None:
            print(f'Output video written to {result}')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        video_colorization = VideoColorizationPipeline(cache_path)
        self.pipeline_inference(video_colorization, self.test_video)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        video_colorization = pipeline(
            task=Tasks.video_colorization, model=self.model_id)
        self.pipeline_inference(video_colorization, self.test_video)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        video_colorization = pipeline(Tasks.video_colorization)
        self.pipeline_inference(video_colorization, self.test_video)

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
