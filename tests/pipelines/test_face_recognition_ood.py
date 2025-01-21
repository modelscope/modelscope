# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import numpy as np

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class FaceRecognitionOodTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.face_recognition
        self.model_id = 'damo/cv_ir_face-recognition-ood_rts'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_face_compare(self):
        img1 = 'data/test/images/face_recognition_1.png'
        img2 = 'data/test/images/face_recognition_2.png'

        face_recognition = pipeline(self.task, model=self.model_id)
        result1 = face_recognition(img1)
        emb1 = result1[OutputKeys.IMG_EMBEDDING]

        result2 = face_recognition(img2)
        emb2 = result2[OutputKeys.IMG_EMBEDDING]

        if emb1 is None or emb2 is None:
            print('No Detected Face.')
        else:
            sim = np.dot(emb1[0], emb2[0])

            score1 = result1[OutputKeys.SCORES][0][0]
            score2 = result2[OutputKeys.SCORES][0][0]

            print(f'Cos similarity={sim:.3f}, img1:{img1}  img2:{img2}')
            print(f'OOD score: img1:{score1:.3f}  img2:{score2:.3f}')


if __name__ == '__main__':
    unittest.main()
