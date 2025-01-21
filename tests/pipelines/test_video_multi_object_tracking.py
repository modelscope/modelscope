# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class MultiObjectTracking(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.video_multi_object_tracking
        self.model_id = 'damo/cv_yolov5_video-multi-object-tracking_fairmot'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_end2end(self):
        video_multi_object_tracking = pipeline(
            Tasks.video_multi_object_tracking, model=self.model_id)
        video_path = 'data/test/videos/MOT17-03-partial.mp4'
        result = video_multi_object_tracking(video_path)
        assert result and (OutputKeys.LABELS in result) and (OutputKeys.BOXES
                                                             in result)
        assert len(result[OutputKeys.LABELS]) == len(result[OutputKeys.BOXES])

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        video_multi_object_tracking = pipeline(
            Tasks.video_multi_object_tracking)
        video_path = 'data/test/videos/MOT17-03-partial.mp4'
        result = video_multi_object_tracking(video_path)
        assert result and (OutputKeys.LABELS in result) and (OutputKeys.BOXES
                                                             in result)
        assert len(result[OutputKeys.LABELS]) == len(result[OutputKeys.BOXES])


if __name__ == '__main__':
    unittest.main()
