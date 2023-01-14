# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

import cv2
import numpy as np

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import depth_to_color
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class PanoramaDepthEstimationTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = 'panorama-depth-estimation'
        self.model_id = 'damo/cv_unifuse_panorama-depth-estimation'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_panorama_depth_estimation(self):
        input_location = 'data/test/images/panorama_depth_estimation.jpg'
        estimator = pipeline(
            Tasks.panorama_depth_estimation, model=self.model_id)
        result = estimator(input_location)
        depth_vis = result[OutputKeys.DEPTHS_COLOR]
        cv2.imwrite('result.jpg', depth_vis)
        print('test_panorama_depth_estimation DONE')


if __name__ == '__main__':
    unittest.main()
