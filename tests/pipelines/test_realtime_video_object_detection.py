# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import cv2
import numpy as np

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import show_video_object_detection_result
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()


class RealtimeVideoObjectDetectionTest(unittest.TestCase,
                                       DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_cspnet_video-object-detection_streamyolo'
        self.test_video = 'data/test/videos/test_realtime_vod.mp4'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        realtime_video_object_detection = pipeline(
            Tasks.video_object_detection, model=self.model_id)
        result = realtime_video_object_detection(self.test_video)
        if result:
            logger.info('Video output to test_vod_results.avi')
            show_video_object_detection_result(self.test_video,
                                               result[OutputKeys.BOXES],
                                               result[OutputKeys.LABELS],
                                               'test_vod_results.avi')
        else:
            raise ValueError('process error')

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
