# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
import unittest

import cv2
import numpy as np

from modelscope.msdatasets import MsDataset
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import draw_face_detection_result
from modelscope.utils.test_utils import test_level


class FacialLandmarkConfidenceTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_manual_facial-landmark-confidence_flcm'

    def show_result(self, img_path, facial_expression_result):
        img = draw_face_detection_result(img_path, facial_expression_result)
        cv2.imwrite('result.png', img)
        print(f'output written to {osp.abspath("result.png")}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        flcm = pipeline(Tasks.face_2d_keypoints, model=self.model_id)
        img_path = 'data/test/images/face_recognition_1.png'
        result = flcm(img_path)
        self.show_result(img_path, result)


if __name__ == '__main__':
    unittest.main()
