# Copyright (c) Alibaba, Inc. and its affiliates.
import pdb
import unittest

import cv2
import numpy as np
from PIL import Image

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class Body3DKeypointsTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_canonical_body-3d-keypoints_video'
        self.test_video = 'data/test/videos/Walking.54138969.mp4'

    def pipeline_inference(self, pipeline: Pipeline, pipeline_input):
        output = pipeline(pipeline_input)
        poses = np.array(output[OutputKeys.POSES])
        print(f'result 3d points shape {poses.shape}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub_with_video_file(self):
        body_3d_keypoints = pipeline(
            Tasks.body_3d_keypoints, model=self.model_id)
        self.pipeline_inference(body_3d_keypoints, self.test_video)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub_with_video_stream(self):
        body_3d_keypoints = pipeline(Tasks.body_3d_keypoints)
        cap = cv2.VideoCapture(self.test_video)
        if not cap.isOpened():
            raise Exception('modelscope error: %s cannot be decoded by OpenCV.'
                            % (self.test_video))
        self.pipeline_inference(body_3d_keypoints, cap)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        body_3d_keypoints = pipeline(Tasks.body_3d_keypoints)
        self.pipeline_inference(body_3d_keypoints, self.test_video)


if __name__ == '__main__':
    unittest.main()
