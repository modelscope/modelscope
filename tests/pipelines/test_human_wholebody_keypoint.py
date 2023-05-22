# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import cv2

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class EasyCVFace2DKeypointsPipelineTest(unittest.TestCase):

    @unittest.skip('skip easycv related cases')
    def test_human_wholebody_keypoint(self):
        img_path = 'data/test/images/keypoints_detect/img_test_wholebody.jpg'
        model_id = 'damo/cv_hrnetw48_human-wholebody-keypoint_image'

        human_wholebody_keypoint_pipeline = pipeline(
            task=Tasks.human_wholebody_keypoint, model=model_id)
        output = human_wholebody_keypoint_pipeline(img_path)

        output_keypoints = output[OutputKeys.KEYPOINTS]
        output_pose = output[OutputKeys.BOXES]

        human_wholebody_keypoint_pipeline.predict_op.show_result(
            img_path,
            output_keypoints,
            output_pose,
            scale=1,
            save_path='human_wholebody_keypoint_ret.jpg')

        for keypoint in output_keypoints:
            self.assertEqual(keypoint.shape[0], 133)
        for box in output_pose:
            self.assertEqual(box.shape[0], 4)


if __name__ == '__main__':
    unittest.main()
