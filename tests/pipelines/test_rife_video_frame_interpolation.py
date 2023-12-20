# Copyright (c) Alibaba, Inc. and its affiliates.
import sys
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.cv import RIFEVideoFrameInterpolationPipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class RIFEVideoFrameInterpolationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.video_frame_interpolation
        self.model_id = 'Damo_XR_Lab/cv_rife_video-frame-interpolation'
        self.test_video = 'data/test/videos/video_frame_interpolation_test.mp4'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        pipeline = RIFEVideoFrameInterpolationPipeline(cache_path)
        pipeline.group_key = self.task
        out_video_path = pipeline(
            input=self.test_video)[OutputKeys.OUTPUT_VIDEO]
        print('pipeline: the output video path is {}'.format(out_video_path))


if __name__ == '__main__':
    unittest.main()
