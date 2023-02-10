# Copyright (c) Alibaba, Inc. and its affiliates.

import datetime
import os.path as osp
import tempfile
from typing import Any, Dict, List, Union

import cv2
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import torch
from matplotlib import animation
from matplotlib.animation import writers
from matplotlib.ticker import MultipleLocator

from modelscope.metainfo import Pipelines
from modelscope.models.cv.body_3d_keypoints.cannonical_pose.body_3d_pose import \
    KeypointsTypes
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Input, Model, Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

matplotlib.use('Agg')

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

    def __init__(self, model: str, **kwargs):
        """Human body 3D pose estimation.

        Args:
            model (str): model id on modelscope hub.
            kwargs (dict, `optional`): Extra kwargs passed into the preprocessor's constructor.
        Example:
            >>> from modelscope.pipelines import pipeline
            >>> body_3d_keypoints = pipeline(Tasks.body_3d_keypoints,
                model='damo/cv_hdformer_body-3d-keypoints_video')
            >>> test_video_url = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/Walking.54138969.mp4'
            >>> output = body_3d_keypoints(test_video_url)
            >>> print(output)
        """
        super().__init__(model=model, **kwargs)

        self.keypoint_model_3d = self.model
        self.keypoint_model_3d.eval()

        # init human body 2D keypoints detection pipeline
        self.human_body_2d_kps_det_pipeline = 'damo/cv_hrnetv2w32_body-2d-keypoints_image'
        self.human_body_2d_kps_detector = pipeline(
            Tasks.body_2d_keypoints,
            model=self.human_body_2d_kps_det_pipeline,
            device='gpu' if torch.cuda.is_available() else 'cpu')

        self.max_frame = self.keypoint_model_3d.cfg.model.INPUT.MAX_FRAME \
            if hasattr(self.keypoint_model_3d.cfg.model.INPUT, 'MAX_FRAME') \
            else self.keypoint_model_3d.cfg.model.INPUT.max_frame  # max video frame number to be predicted 3D joints

    def preprocess(self, input: Input) -> Dict[str, Any]:
        self.video_url = input
        video_frames = self.read_video_frames(self.video_url)
        if 0 == len(video_frames):
            res = {'success': False, 'msg': 'get video frame failed.'}
            return res

        all_2d_poses = []
        all_boxes_with_socre = []
        for i, frame in enumerate(video_frames):
            kps_2d = self.human_body_2d_kps_detector(frame)
            if [] == kps_2d.get('boxes'):
                res = {
                    'success': False,
                    'msg': f'fail to detect person at image frame {i}'
                }
                return res

            box = kps_2d['boxes'][
                0]  # box: [[[x1, y1], [x2, y2]]], N human boxes per frame, [0] represent using first detected bbox
            pose = kps_2d['keypoints'][0]  # keypoints: [15, 2]
            score = kps_2d['scores'][0]  # keypoints: [15, 2]
            all_2d_poses.append(pose)
            all_boxes_with_socre.append(
                list(np.array(box).reshape(
                    (-1))) + [score])  # construct to list with shape [5]
            if (i + 1) >= self.max_frame:
                break

        all_2d_poses_np = np.array(all_2d_poses).reshape(
            (len(all_2d_poses), 15,
             2))  # 15: 2d keypoints number, 2: keypoint coordinate (x, y)
        all_boxes_np = np.array(all_boxes_with_socre).reshape(
            (len(all_boxes_with_socre), 5))  # [x1, y1, x2, y2, score]

        joint_num = self.keypoint_model_3d.cfg.model.MODEL.IN_NUM_JOINTS \
            if hasattr(self.keypoint_model_3d.cfg.model.MODEL, 'IN_NUM_JOINTS') \
            else self.keypoint_model_3d.cfg.model.MODEL.n_joints
        kps_2d_h36m_17 = convert_2_h36m_data(
            all_2d_poses_np, all_boxes_np, joints_nbr=joint_num)
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
        output_video_path = kwargs.get('output_video', None)
        if output_video_path is None:
            output_video_path = tempfile.NamedTemporaryFile(suffix='.mp4').name

        res = {
            OutputKeys.KEYPOINTS: [],
            OutputKeys.TIMESTAMPS: [],
            OutputKeys.OUTPUT_VIDEO: output_video_path
        }

        if not input['success']:
            res[OutputKeys.OUTPUT_VIDEO] = self.video_url
        else:
            poses = input[KeypointsTypes.POSES_CAMERA]
            pred_3d_pose = poses.data.cpu().numpy()[
                0]  # [frame_num, joint_num, joint_dim]

            if 'render' in self.keypoint_model_3d.cfg.keys():
                self.render_prediction(pred_3d_pose, output_video_path)
                res[OutputKeys.OUTPUT_VIDEO] = output_video_path

            res[OutputKeys.KEYPOINTS] = pred_3d_pose
            res[OutputKeys.TIMESTAMPS] = self.timestamps
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

        def timestamp_format(seconds):
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            time = '%02d:%02d:%06.3f' % (h, m, s)
            return time

        frames = []
        self.timestamps = []  # for video render
        if isinstance(video_url, str):
            cap = cv2.VideoCapture(video_url)
            if not cap.isOpened():
                raise Exception(
                    'modelscope error: %s cannot be decoded by OpenCV.' %
                    (video_url))
        else:
            cap = video_url

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps is None or self.fps <= 0:
            raise Exception('modelscope error: %s cannot get video fps info.' %
                            (video_url))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.timestamps.append(
                timestamp_format(seconds=frame_idx / self.fps))
            frame_idx += 1
            frames.append(frame)
            if frame_idx >= self.max_frame:
                break
        cap.release()
        return frames

    def render_prediction(self, pose3d_cam_rr, output_video_path):
        """render predict result 3d poses.

        Args:
            pose3d_cam_rr (nd.array): [frame_num, joint_num, joint_dim], 3d pose joints
            output_video_path (str): output path for video
        Returns:
        """
        frame_num = pose3d_cam_rr.shape[0]

        left_points = [11, 12, 13, 4, 5, 6]  # joints of left body
        edges = [[0, 1], [0, 4], [0, 7], [1, 2], [4, 5], [5, 6], [2,
                                                                  3], [7, 8],
                 [8, 9], [8, 11], [8, 14], [14, 15], [15, 16], [11, 12],
                 [12, 13], [9, 10]]  # connection between joints

        fig = plt.figure()
        ax = p3.Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        x_major_locator = MultipleLocator(0.5)

        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(x_major_locator)
        ax.zaxis.set_major_locator(x_major_locator)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        # view direction
        azim = self.keypoint_model_3d.cfg.render.azim
        elev = self.keypoint_model_3d.cfg.render.elev
        ax.view_init(elev, azim)

        # init plot, essentially
        x = pose3d_cam_rr[0, :, 0]
        y = pose3d_cam_rr[0, :, 1]
        z = pose3d_cam_rr[0, :, 2]
        points, = ax.plot(x, y, z, 'r.')

        def renderBones(xs, ys, zs):
            """render bones in skeleton

            Args:
                xs (nd.array): [joint_num, joint_channel]
                ys (nd.array): [joint_num, joint_channel]
                zs (nd.array): [joint_num, joint_channel]
            """
            bones = {}
            for idx, edge in enumerate(edges):
                index1, index2 = edge[0], edge[1]
                if index1 in left_points:
                    edge_color = 'red'
                else:
                    edge_color = 'blue'
                connect = ax.plot([xs[index1], xs[index2]],
                                  [ys[index1], ys[index2]],
                                  [zs[index1], zs[index2]],
                                  linewidth=2,
                                  color=edge_color)  # plot edge
                bones[idx] = connect[0]
            return bones

        bones = renderBones(x, y, z)

        def update(frame_idx, points, bones):
            """update animation

            Args:
                frame_idx (int): frame index
                points (mpl_toolkits.mplot3d.art3d.Line3D): skeleton points ploter
                bones (dict[int, mpl_toolkits.mplot3d.art3d.Line3D]): connection ploter

            Returns:
                tuple: points and bones ploter
            """
            xs = pose3d_cam_rr[frame_idx, :, 0]
            ys = pose3d_cam_rr[frame_idx, :, 1]
            zs = pose3d_cam_rr[frame_idx, :, 2]

            # update bones
            for idx, edge in enumerate(edges):
                index1, index2 = edge[0], edge[1]
                x1x2 = (xs[index1], xs[index2])
                y1y2 = (ys[index1], ys[index2])
                z1z2 = (zs[index1], zs[index2])
                bones[idx].set_xdata(x1x2)
                bones[idx].set_ydata(y1y2)
                bones[idx].set_3d_properties(z1z2, 'z')

            # update joints
            points.set_data(xs, ys)
            points.set_3d_properties(zs, 'z')
            if 0 == frame_idx / 100:
                logger.info(f'rendering {frame_idx}/{frame_num}')
            return points, bones

        ani = animation.FuncAnimation(
            fig=fig,
            func=update,
            frames=frame_num,
            interval=self.fps,
            fargs=(points, bones))

        # save mp4
        Writer = writers['ffmpeg']
        writer = Writer(fps=self.fps, metadata={}, bitrate=4096)
        ani.save(output_video_path, writer=writer)
