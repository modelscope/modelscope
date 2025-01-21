# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import cv2
from PIL import Image

from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import draw_keypoints
from modelscope.utils.test_utils import test_level


class Body2DKeypointsTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.body_2d_keypoints
        self.model_id = 'damo/cv_hrnetv2w32_body-2d-keypoints_image'
        self.test_image = 'data/test/images/keypoints_detect/000000438862.jpg'

    def pipeline_inference(self, pipeline: Pipeline, pipeline_input):
        output = pipeline(pipeline_input)
        image = draw_keypoints(output, self.test_image)
        cv2.imwrite('pose_keypoint.jpg', image)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_modelhub_with_image_file(self):
        body_2d_keypoints = pipeline(self.task, model=self.model_id)
        self.pipeline_inference(body_2d_keypoints, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub_with_image_input(self):
        body_2d_keypoints = pipeline(self.task, model=self.model_id)
        self.pipeline_inference(body_2d_keypoints, Image.open(self.test_image))


if __name__ == '__main__':
    unittest.main()
