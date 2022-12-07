# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
import unittest

import cv2
import numpy as np

from modelscope.msdatasets import MsDataset
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import draw_face_attribute_result
from modelscope.utils.test_utils import test_level


class FaceAttributeRecognitionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_resnet34_face-attribute-recognition_fairface'

    def show_result(self, img_path, facial_expression_result):
        img = draw_face_attribute_result(img_path, facial_expression_result)
        cv2.imwrite('result.png', img)
        print(f'output written to {osp.abspath("result.png")}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        fair_face = pipeline(
            Tasks.face_attribute_recognition, model=self.model_id)
        img_path = 'data/test/images/face_recognition_1.png'
        result = fair_face(img_path)
        self.show_result(img_path, result)


if __name__ == '__main__':
    unittest.main()
