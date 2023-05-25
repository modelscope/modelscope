# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import numpy as np

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class FmFaceRecognitionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.face_recognition
        self.model_id = 'damo/cv_manual_face-recognition_frfm'

    # @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    @unittest.skip(
        'currently this model has been unavailable due to accuracy problem.')
    def test_face_compare(self):
        img1 = 'data/test/images/face_recognition_1.png'
        img2 = 'data/test/images/face_recognition_2.png'

        face_recognition = pipeline(
            Tasks.face_recognition, model=self.model_id)
        emb1 = face_recognition(img1)[OutputKeys.IMG_EMBEDDING]
        emb2 = face_recognition(img2)[OutputKeys.IMG_EMBEDDING]
        if emb1 is None or emb2 is None:
            print('No Detected Face.')
        else:
            sim = np.dot(emb1[0], emb2[0])
            print(f'Cos similarity={sim:.3f}, img1:{img1}  img2:{img2}')


if __name__ == '__main__':
    unittest.main()
