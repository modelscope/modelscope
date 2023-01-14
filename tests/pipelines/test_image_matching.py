# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest
from pathlib import Path

import cv2
import matplotlib.cm as cm
import numpy as np

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import match_pair_visualization
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class ImageMatchingTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = 'image-matching'
        self.model_id = 'damo/cv_quadtree_attention_image-matching_outdoor'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_image_matching(self):
        input_location = [[
            'data/test/images/image_matching1.jpg',
            'data/test/images/image_matching2.jpg'
        ]]
        estimator = pipeline(Tasks.image_matching, model=self.model_id)
        result = estimator(input_location)
        kpts0, kpts1, conf = result[0][OutputKeys.MATCHES]

        match_pair_visualization(
            input_location[0][0],
            input_location[0][1],
            kpts0,
            kpts1,
            conf,
            output_filename='quadtree_match.png')

        print('test_image_matching DONE')


if __name__ == '__main__':
    unittest.main()
