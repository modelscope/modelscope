# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path
import unittest

import cv2

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class HumanNormalEstimationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = 'human-normal-estimation'
        self.model_id = 'Damo_XR_Lab/cv_human_monocular-normal-estimation'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_image_normal_estimation(self):
        cur_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        input_location = f'{cur_dir}/data/test/images/human_normal_estimation.png'
        estimator = pipeline(
            Tasks.human_normal_estimation, model=self.model_id)
        result = estimator(input_location)
        normals_vis = result[OutputKeys.NORMALS_COLOR]

        input_img = cv2.imread(input_location)
        normals_vis = cv2.resize(
            normals_vis, dsize=(input_img.shape[1], input_img.shape[0]))
        cv2.imwrite('result.jpg', normals_vis)


if __name__ == '__main__':
    unittest.main()
