# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
import tempfile
import unittest

import cv2
import numpy as np

from modelscope.fileio import File
from modelscope.msdatasets import MsDataset
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.test_utils import test_level


class FaceRecognitionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.recog_model_id = 'damo/cv_ir101_facerecognition_cfglint'
        self.det_model_id = 'damo/cv_resnet_facedetection_scrfd10gkps'

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_face_compare(self):
        img1 = 'data/test/images/face_recognition_1.png'
        img2 = 'data/test/images/face_recognition_2.png'

        face_detection = pipeline(
            Tasks.face_detection, model=self.det_model_id)
        face_recognition = pipeline(
            Tasks.face_recognition,
            face_detection=face_detection,
            model=self.recog_model_id)
        # note that for dataset output, the inference-output is a Generator that can be iterated.
        emb1 = face_recognition(img1)[OutputKeys.IMG_EMBEDDING]
        emb2 = face_recognition(img2)[OutputKeys.IMG_EMBEDDING]
        sim = np.dot(emb1[0], emb2[0])
        print(f'Cos similarity={sim:.3f}, img1:{img1}  img2:{img2}')


if __name__ == '__main__':
    unittest.main()
