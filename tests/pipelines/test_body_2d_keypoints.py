# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import cv2
import numpy as np

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

lst_parent_ids_17 = [0, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14]
lst_left_ids_17 = [1, 3, 5, 7, 9, 11, 13, 15]
lst_right_ids_17 = [2, 4, 6, 8, 10, 12, 14, 16]
lst_spine_ids_17 = [0]

lst_parent_ids_15 = [0, 0, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 1]
lst_left_ids_15 = [2, 3, 4, 8, 9, 10]
lst_right_ids_15 = [5, 6, 7, 11, 12, 13]
lst_spine_ids_15 = [0, 1, 14]


def draw_joints(image, np_kps, score, threshold=0.2):
    if np_kps.shape[0] == 17:
        lst_parent_ids = lst_parent_ids_17
        lst_left_ids = lst_left_ids_17
        lst_right_ids = lst_right_ids_17

    elif np_kps.shape[0] == 15:
        lst_parent_ids = lst_parent_ids_15
        lst_left_ids = lst_left_ids_15
        lst_right_ids = lst_right_ids_15

    for i in range(len(lst_parent_ids)):
        pid = lst_parent_ids[i]
        if i == pid:
            continue

        if (score[i] < threshold or score[1] < threshold):
            continue

        if i in lst_left_ids and pid in lst_left_ids:
            color = (0, 255, 0)
        elif i in lst_right_ids and pid in lst_right_ids:
            color = (255, 0, 0)
        else:
            color = (0, 255, 255)

        cv2.line(image, (int(np_kps[i, 0]), int(np_kps[i, 1])),
                 (int(np_kps[pid][0]), int(np_kps[pid, 1])), color, 3)

    for i in range(np_kps.shape[0]):
        if score[i] < threshold:
            continue
        cv2.circle(image, (int(np_kps[i, 0]), int(np_kps[i, 1])), 5,
                   (0, 0, 255), -1)


def draw_box(image, box):
    cv2.rectangle(image, (int(box[0][0]), int(box[0][1])),
                  (int(box[1][0]), int(box[1][1])), (0, 0, 255), 2)


class Body2DKeypointsTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_hrnetv2w32_body-2d-keypoints_image'
        self.test_image = 'data/test/images/keypoints_detect/000000438862.jpg'

    def pipeline_inference(self, pipeline: Pipeline):
        output = pipeline(self.test_image)
        poses = np.array(output[OutputKeys.POSES])
        scores = np.array(output[OutputKeys.SCORES])
        boxes = np.array(output[OutputKeys.BOXES])
        assert len(poses) == len(scores) and len(poses) == len(boxes)
        image = cv2.imread(self.test_image, -1)
        for i in range(len(poses)):
            draw_box(image, np.array(boxes[i]))
            draw_joints(image, np.array(poses[i]), np.array(scores[i]))
        cv2.imwrite('pose_keypoint.jpg', image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        body_2d_keypoints = pipeline(
            Tasks.body_2d_keypoints, model=self.model_id)
        self.pipeline_inference(body_2d_keypoints)


if __name__ == '__main__':
    unittest.main()
