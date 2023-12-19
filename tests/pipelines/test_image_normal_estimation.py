# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

import cv2
import numpy as np

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ImageNormalEstimationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = 'image-normal-estimation'
        self.model_id = 'Damo_XR_Lab/cv_omnidata_image-normal-estimation_normal'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_image_normal_estimation(self):
        input_location = 'data/test/images/image_normal_estimation.jpg'
        estimator = pipeline(
            Tasks.image_normal_estimation, model=self.model_id)
        result = estimator(input_location)
        normals_vis = result[OutputKeys.NORMALS_COLOR]
        cv2.imwrite('result.jpg', normals_vis[:, :, ::-1])

        print('test_image_normal_estimation DONE')


if __name__ == '__main__':
    unittest.main()
