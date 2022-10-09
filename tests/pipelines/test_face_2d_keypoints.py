# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import cv2

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class EasyCVFace2DKeypointsPipelineTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_face_2d_keypoints(self):
        img_path = 'data/test/images/keypoints_detect/test_img_face_2d_keypoints.png'
        model_id = 'damo/cv_mobilenet_face-2d-keypoints_alignment'

        face_2d_keypoints_align = pipeline(
            task=Tasks.face_2d_keypoints, model=model_id)
        output = face_2d_keypoints_align(img_path)[0]

        output_keypoints = output[OutputKeys.KEYPOINTS]
        output_pose = output[OutputKeys.POSES]

        img = cv2.imread(img_path)
        img = face_2d_keypoints_align.show_result(
            img, output_keypoints, scale=2, save_path='face_keypoints.jpg')

        self.assertEqual(output_keypoints.shape[0], 106)
        self.assertEqual(output_keypoints.shape[1], 2)
        self.assertEqual(output_pose.shape[0], 3)


if __name__ == '__main__':
    unittest.main()
