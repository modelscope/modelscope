# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
import unittest

import cv2

from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import draw_face_detection_result
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class FaceDetectionTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.face_detection
        self.model_id = 'damo/cv_resnet_facedetection_scrfd10gkps'

    def show_result(self, img_path, detection_result):
        img = draw_face_detection_result(img_path, detection_result)
        cv2.imwrite('result.png', img)
        print(f'output written to {osp.abspath("result.png")}')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_dataset(self):
        input_location = ['data/test/images/face_detection2.jpeg']

        dataset = MsDataset.load(input_location, target='image')
        face_detection = pipeline(Tasks.face_detection, model=self.model_id)
        # note that for dataset output, the inference-output is a Generator that can be iterated.
        result = face_detection(dataset)
        result = next(result)
        self.show_result(input_location[0], result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        face_detection = pipeline(Tasks.face_detection, model=self.model_id)
        img_path = 'data/test/images/face_detection2.jpeg'

        result = face_detection(img_path)
        self.show_result(img_path, result)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        face_detection = pipeline(Tasks.face_detection)
        img_path = 'data/test/images/face_detection2.jpeg'
        result = face_detection(img_path)
        self.show_result(img_path, result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
