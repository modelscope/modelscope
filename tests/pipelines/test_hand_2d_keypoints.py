# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class Hand2DKeypointsPipelineTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_hand_2d_keypoints(self):
        img_path = 'data/test/images/hand_keypoints.jpg'
        model_id = 'damo/cv_hrnetw18_hand-pose-keypoints_coco-wholebody'

        hand_keypoint = pipeline(task=Tasks.hand_2d_keypoints, model=model_id)
        outputs = hand_keypoint(img_path)
        self.assertEqual(len(outputs), 1)

        results = outputs[0]
        self.assertIn(OutputKeys.KEYPOINTS, results.keys())
        self.assertIn(OutputKeys.BOXES, results.keys())
        self.assertEqual(results[OutputKeys.KEYPOINTS].shape[1], 21)
        self.assertEqual(results[OutputKeys.KEYPOINTS].shape[2], 3)
        self.assertEqual(results[OutputKeys.BOXES].shape[1], 4)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_hand_2d_keypoints_with_default_model(self):
        img_path = 'data/test/images/hand_keypoints.jpg'

        hand_keypoint = pipeline(task=Tasks.hand_2d_keypoints)
        outputs = hand_keypoint(img_path)
        self.assertEqual(len(outputs), 1)

        results = outputs[0]
        self.assertIn(OutputKeys.KEYPOINTS, results.keys())
        self.assertIn(OutputKeys.BOXES, results.keys())
        self.assertEqual(results[OutputKeys.KEYPOINTS].shape[1], 21)
        self.assertEqual(results[OutputKeys.KEYPOINTS].shape[2], 3)
        self.assertEqual(results[OutputKeys.BOXES].shape[1], 4)


if __name__ == '__main__':
    unittest.main()
