# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
import unittest

import cv2

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import draw_face_detection_no_lm_result
from modelscope.utils.test_utils import test_level


class FaceQualityAssessmentTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_manual_face-quality-assessment_fqa'
        self.img_path = 'data/test/images/vision_efficient_tuning_test_sunflower.jpg'
        self.img_path = 'data/test/images/face_recognition_1.png'

    def show_result(self, img_path, detection_result):
        img = draw_face_detection_no_lm_result(img_path, detection_result)
        cv2.imwrite('result.png', img)
        print(f'output written to {osp.abspath("result.png")}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        face_quality_assessment = pipeline(
            Tasks.face_quality_assessment, model=self.model_id)
        result = face_quality_assessment(self.img_path)
        if result[OutputKeys.SCORES] is None:
            print('No Detected Face.')
        else:
            self.show_result(self.img_path, result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_default_model(self):
        face_quality_assessment = pipeline(Tasks.face_quality_assessment)
        result = face_quality_assessment(self.img_path)
        if result[OutputKeys.SCORES] is None:
            print('No Detected Face.')
        else:
            self.show_result(self.img_path, result)


if __name__ == '__main__':
    unittest.main()
