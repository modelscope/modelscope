# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
import unittest

import cv2

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import draw_face_detection_no_lm_result
from modelscope.utils.test_utils import test_level


class MogFaceDetectionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_resnet101_face-detection_cvpr22papermogface'

    def show_result(self, img_path, detection_result):
        img = draw_face_detection_no_lm_result(img_path, detection_result)
        cv2.imwrite('result.png', img)
        print(f'output written to {osp.abspath("result.png")}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        face_detection = pipeline(Tasks.face_detection, model=self.model_id)
        img_path = 'data/test/images/mog_face_detection.jpg'

        result = face_detection(img_path)
        self.show_result(img_path, result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_default_model(self):
        face_detection = pipeline(Tasks.face_detection)
        img_path = 'data/test/images/mog_face_detection.jpg'

        result = face_detection(img_path)
        self.show_result(img_path, result)


if __name__ == '__main__':
    unittest.main()
