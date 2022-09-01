import os
import os.path as osp
from typing import Any, Dict, List, Union

import cv2
import numpy as np
import torch

from modelscope.metainfo import Pipelines
from modelscope.models.cv.body_3d_keypoints.body_3d_pose import (
    BodyKeypointsDetection3D, KeypointsTypes)
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Input, Model, Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


def convert_2_h36m(joints, joints_nbr=15):
    lst_mappings = [[0, 8], [1, 7], [2, 12], [3, 13], [4, 14], [5, 9], [6, 10],
                    [7, 11], [8, 1], [9, 2], [10, 3], [11, 4], [12, 5],
                    [13, 6], [14, 0]]
    nbr, dim = joints.shape
    h36m_joints = np.zeros((nbr, dim))
    for mapping in lst_mappings:
        h36m_joints[mapping[1]] = joints[mapping[0]]

    if joints_nbr == 17:
        lst_mappings_17 = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4],
                                    [5, 5], [6, 6], [7, 8], [8, 10], [9, 11],
                                    [10, 12], [11, 13], [12, 14], [13, 15],
                                    [14, 16]])
        h36m_joints_17 = np.zeros((17, 2))
        h36m_joints_17[lst_mappings_17[:, 1]] = h36m_joints[lst_mappings_17[:,
                                                                            0]]
        h36m_joints_17[7] = (h36m_joints_17[0] + h36m_joints_17[8]) * 0.5
        h36m_joints_17[9] = (h36m_joints_17[8] + h36m_joints_17[10]) * 0.5
        h36m_joints = h36m_joints_17

    return h36m_joints


def smooth_pts(cur_pts, pre_pts, bbox, smooth_x=15.0, smooth_y=15.0):
    if pre_pts is None:
        return cur_pts

    w, h = bbox[1] - bbox[0]
    if w == 0 or h == 0:
        return cur_pts

    size_pre = len(pre_pts)
    size_cur = len(cur_pts)
    if (size_pre == 0 or size_cur == 0):
        return cur_pts

    factor_x = -(smooth_x / w)
    factor_y = -(smooth_y / w)

    for i in range(size_cur):
        w_x = np.exp(factor_x * np.abs(cur_pts[i][0] - pre_pts[i][0]))
        w_y = np.exp(factor_y * np.abs(cur_pts[i][1] - pre_pts[i][1]))
        cur_pts[i][0] = (1.0 - w_x) * cur_pts[i][0] + w_x * pre_pts[i][0]
        cur_pts[i][1] = (1.0 - w_y) * cur_pts[i][1] + w_y * pre_pts[i][1]
    return cur_pts


def smoothing(lst_kps, lst_bboxes, smooth_x=15.0, smooth_y=15.0):
    assert lst_kps.shape[0] == lst_bboxes.shape[0]

    lst_smoothed_kps = []
    prev_pts = None
    for i in range(lst_kps.shape[0]):
        smoothed_cur_kps = smooth_pts(lst_kps[i], prev_pts,
                                      lst_bboxes[i][0:-1].reshape(2, 2),
                                      smooth_x, smooth_y)
        lst_smoothed_kps.append(smoothed_cur_kps)
        prev_pts = smoothed_cur_kps

    return np.array(lst_smoothed_kps)


def convert_2_h36m_data(lst_kps, lst_bboxes, joints_nbr=15):
    lst_kps = lst_kps.squeeze()
    lst_bboxes = lst_bboxes.squeeze()

    assert lst_kps.shape[0] == lst_bboxes.shape[0]

    lst_kps = smoothing(lst_kps, lst_bboxes)

    keypoints = []
    for i in range(lst_kps.shape[0]):
        h36m_joints_2d = convert_2_h36m(lst_kps[i], joints_nbr=joints_nbr)
        keypoints.append(h36m_joints_2d)
    return keypoints


@PIPELINES.register_module(
    Tasks.body_3d_keypoints, module_name=Pipelines.body_3d_keypoints)
class Body3DKeypointsPipeline(Pipeline):

    def __init__(self, model: Union[str, BodyKeypointsDetection3D], **kwargs):
        """Human body 3D pose estimation.

        Args:
            model (Union[str, BodyKeypointsDetection3D]): model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)

        self.keypoint_model_3d = model if isinstance(
            model, BodyKeypointsDetection3D) else Model.from_pretrained(model)
        self.keypoint_model_3d.eval()

        # init human body 2D keypoints detection pipeline
        self.human_body_2d_kps_det_pipeline = 'damo/cv_hrnetv2w32_body-2d-keypoints_image'
        self.human_body_2d_kps_detector = pipeline(
            Tasks.body_2d_keypoints,
            model=self.human_body_2d_kps_det_pipeline,
            device='gpu' if torch.cuda.is_available() else 'cpu')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        video_frames = self.read_video_frames(input)
        if 0 == len(video_frames):
            res = {'success': False, 'msg': 'get video frame failed.'}
            return res

        all_2d_poses = []
        all_boxes_with_socre = []
        max_frame = self.keypoint_model_3d.cfg.model.INPUT.MAX_FRAME  # max video frame number to be predicted 3D joints
        for i, frame in enumerate(video_frames):
            kps_2d = self.human_body_2d_kps_detector(frame)
            box = kps_2d['boxes'][
                0]  # box: [[[x1, y1], [x2, y2]]], N human boxes per frame, [0] represent using first detected bbox
            pose = kps_2d['poses'][0]  # keypoints: [15, 2]
            score = kps_2d['scores'][0]  # keypoints: [15, 2]
            all_2d_poses.append(pose)
            all_boxes_with_socre.append(
                list(np.array(box).reshape(
                    (-1))) + [score])  # construct to list with shape [5]
            if (i + 1) >= max_frame:
                break

        all_2d_poses_np = np.array(all_2d_poses).reshape(
            (len(all_2d_poses), 15,
             2))  # 15: 2d keypoints number, 2: keypoint coordinate (x, y)
        all_boxes_np = np.array(all_boxes_with_socre).reshape(
            (len(all_boxes_with_socre), 5))  # [x1, y1, x2, y2, score]

        kps_2d_h36m_17 = convert_2_h36m_data(
            all_2d_poses_np,
            all_boxes_np,
            joints_nbr=self.keypoint_model_3d.cfg.model.MODEL.IN_NUM_JOINTS)
        kps_2d_h36m_17 = np.array(kps_2d_h36m_17)
        res = {'success': True, 'input_2d_pts': kps_2d_h36m_17}
        return res

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if not input['success']:
            res = {'success': False, 'msg': 'preprocess failed.'}
            return res

        input_2d_pts = input['input_2d_pts']
        outputs = self.keypoint_model_3d.preprocess(input_2d_pts)
        outputs = self.keypoint_model_3d.forward(outputs)
        res = dict({'success': True}, **outputs)
        return res

    def postprocess(self, input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        res = {OutputKeys.POSES: []}

        if not input['success']:
            pass
        else:
            poses = input[KeypointsTypes.POSES_CAMERA]
            res = {OutputKeys.POSES: poses.data.cpu().numpy()}
        return res

    def read_video_frames(self, video_url: Union[str, cv2.VideoCapture]):
        """Read video from local video file or from a video stream URL.

        Args:
            video_url (str or cv2.VideoCapture): Video path or video stream.

        Raises:
            Exception: Open video fail.

        Returns:
            [nd.array]: List of video frames.
        """
        frames = []
        if isinstance(video_url, str):
            cap = cv2.VideoCapture(video_url)
            if not cap.isOpened():
                raise Exception(
                    'modelscope error: %s cannot be decoded by OpenCV.' %
                    (video_url))
        else:
            cap = video_url

        max_frame_num = self.keypoint_model_3d.cfg.model.INPUT.MAX_FRAME
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            frames.append(frame)
            if frame_idx >= max_frame_num:
                break
        cap.release()
        return frames
