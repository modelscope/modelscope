# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import show_video_tracking_result
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class SingleObjectTracking(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.video_single_object_tracking
        self.model_id = 'damo/cv_vitb_video-single-object-tracking_ostrack'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_end2end(self):
        video_single_object_tracking = pipeline(
            Tasks.video_single_object_tracking, model=self.model_id)
        video_path = 'data/test/videos/dog.avi'
        init_bbox = [414, 343, 514, 449]  # [x1, y1, x2, y2]
        result = video_single_object_tracking((video_path, init_bbox))
        print('result is : ', result[OutputKeys.BOXES])
        show_video_tracking_result(video_path, result[OutputKeys.BOXES],
                                   './tracking_result.avi')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        video_single_object_tracking = pipeline(
            Tasks.video_single_object_tracking)
        video_path = 'data/test/videos/dog.avi'
        init_bbox = [414, 343, 514, 449]  # [x1, y1, x2, y2]
        result = video_single_object_tracking((video_path, init_bbox))
        print('result is : ', result[OutputKeys.BOXES])

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
