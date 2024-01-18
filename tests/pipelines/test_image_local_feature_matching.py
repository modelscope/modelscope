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
from modelscope.utils.test_utils import test_level


class ImageLocalFeatureMatchingTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = 'image-local-feature-matching'
        self.model_id = 'Damo_XR_Lab/cv_resnet-transformer_local-feature-matching_outdoor-data'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_image_local_feature_matching(self):
        input_location = [[
            'data/test/images/image_matching1.jpg',
            'data/test/images/image_matching2.jpg'
        ]]
        estimator = pipeline(Tasks.image_local_feature_matching, model=self.model_id)
        result = estimator(input_location)
        kpts0, kpts1, conf = result[0][OutputKeys.MATCHES]
        vis_img = result[0][OutputKeys.OUTPUT_IMG]
        cv2.imwrite("vis_demo.jpg", vis_img)

        print('test_image_local_feature_matching DONE')


if __name__ == '__main__':
    unittest.main()
