# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import show_video_depth_estimation_result
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class VideoDepthEstimationTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = 'video-depth-estimation'
        self.model_id = 'damo/cv_dro-resnet18_video-depth-estimation_indoor'

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_image_depth_estimation(self):
        input_location = 'data/test/videos/video_depth_estimation.mp4'
        estimator = pipeline(Tasks.video_depth_estimation, model=self.model_id)
        result = estimator(input_location)
        show_video_depth_estimation_result(result[OutputKeys.DEPTHS_COLOR],
                                           'out.mp4')

        print('test_video_depth_estimation DONE')


if __name__ == '__main__':

    unittest.main()
