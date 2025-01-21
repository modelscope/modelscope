# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import draw_106face_keypoints
from modelscope.utils.test_utils import test_level


class EasyCVFace2DKeypointsPipelineTest(unittest.TestCase):

    @unittest.skip('skip easycv related cases')
    def test_face_2d_keypoints(self):
        img_path = 'data/test/images/face_detection.png'
        model_id = 'damo/cv_mobilenet_face-2d-keypoints_alignment'

        face_2d_keypoints_align = pipeline(
            task=Tasks.face_2d_keypoints, model=model_id)
        output = face_2d_keypoints_align(img_path)

        output_keypoints = output[OutputKeys.KEYPOINTS]
        output_poses = output[OutputKeys.POSES]
        output_boxes = output[OutputKeys.BOXES]

        draw_106face_keypoints(
            img_path,
            output_keypoints,
            output_boxes,
            scale=2,
            save_path='face_keypoints.jpg')

        for idx in range(len(output_keypoints)):
            self.assertEqual(output_keypoints[idx].shape[0], 106)
            self.assertEqual(output_keypoints[idx].shape[1], 2)
            self.assertEqual(output_poses[idx].shape[0], 3)
            self.assertEqual(output_boxes[idx].shape[0], 4)


if __name__ == '__main__':
    unittest.main()
