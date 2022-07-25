import sys
import unittest

import cv2
import numpy as np

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class VirtualTryonTest(unittest.TestCase):
    model_id = 'damo/cv_daflow_virtual-tryon_base'
    input_imgs = {
        'masked_model': 'data/test/images/virtual_tryon_model.jpg',
        'pose': 'data/test/images/virtual_tryon_pose.jpg',
        'cloth': 'data/test/images/virtual_tryon_cloth.jpg'
    }

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_virtual_tryon = pipeline(
            task=Tasks.virtual_tryon, model=self.model_id)
        img = pipeline_virtual_tryon(self.input_imgs)[OutputKeys.OUTPUT_IMG]
        cv2.imwrite('demo.jpg', img[:, :, ::-1])

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_model_name_default_model(self):
        pipeline_virtual_tryon = pipeline(task=Tasks.virtual_tryon)
        img = pipeline_virtual_tryon(self.input_imgs)[OutputKeys.OUTPUT_IMG]
        cv2.imwrite('demo.jpg', img[:, :, ::-1])


if __name__ == '__main__':
    unittest.main()
