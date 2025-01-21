# Copyright (c) Alibaba, Inc. and its affiliates.
import sys
import unittest

import cv2

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class ImageLayoutEstimationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.indoor_layout_estimation
        self.model_id = 'damo/cv_panovit_indoor-layout-estimation'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_image_layout_estimation(self):
        input_location = 'data/test/images/indoor_layout_estimation.png'
        estimator = pipeline(
            Tasks.indoor_layout_estimation, model=self.model_id)
        result = estimator(input_location)
        layout = result[OutputKeys.LAYOUT]
        cv2.imwrite('layout.jpg', layout)

        print('test_image_layout_estimation DONE')


if __name__ == '__main__':
    unittest.main()
