# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
import os.path as osp
from typing import Any, Dict, List, Union

import numpy as np
import torch

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.body_3d_keypoints.canonical_pose_modules import (
    TemporalModel, TransCan3Dkeys)
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['BodyKeypointsDetection3D']


class KeypointsTypes(object):
    POSES_CAMERA = 'poses_camera'
    POSES_TRAJ = 'poses_traj'


@MODELS.register_module(
    Tasks.body_3d_keypoints, module_name=Models.body_3d_keypoints)
class BodyKeypointsDetection3D(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):

        super().__init__(model_dir, *args, **kwargs)

        self.model_dir = model_dir
        model_path = osp.join(self.model_dir, ModelFile.TORCH_MODEL_FILE)
        cfg_path = osp.join(self.model_dir, ModelFile.CONFIGURATION)
        self.cfg = Config.from_file(cfg_path)
        self._create_model()

        if not osp.exists(model_path):
            raise IOError(f'{model_path} is not exists.')

        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        self.pretrained_state_dict = torch.load(
            model_path, map_location=self._device)

        self.load_pretrained()
        self.to_device(self._device)
        self.eval()

    def _create_model(self):
        self.model_pos = TemporalModel(
            self.cfg.model.MODEL.IN_NUM_JOINTS,
            self.cfg.model.MODEL.IN_2D_FEATURE,
            self.cfg.model.MODEL.OUT_NUM_JOINTS,
            filter_widths=self.cfg.model.MODEL.FILTER_WIDTHS,
            causal=self.cfg.model.MODEL.CAUSAL,
            dropout=self.cfg.model.MODEL.DROPOUT,
            channels=self.cfg.model.MODEL.CHANNELS,
            dense=self.cfg.model.MODEL.DENSE)

        receptive_field = self.model_pos.receptive_field()
        self.pad = (receptive_field - 1) // 2
        if self.cfg.model.MODEL.CAUSAL:
            self.causal_shift = self.pad
        else:
            self.causal_shift = 0

        self.model_traj = TransCan3Dkeys(
            in_channels=self.cfg.model.MODEL.IN_NUM_JOINTS
            * self.cfg.model.MODEL.IN_2D_FEATURE,
            num_features=1024,
            out_channels=self.cfg.model.MODEL.OUT_3D_FEATURE,
            num_blocks=4,
            time_window=receptive_field)

    def eval(self):
        self.model_pos.eval()
        self.model_traj.eval()

    def train(self):
        self.model_pos.train()
        self.model_traj.train()

    def to_device(self, device):
        self.model_pos = self.model_pos.to(device)
        self.model_traj = self.model_traj.to(device)

    def load_pretrained(self):
        if 'model_pos' in self.pretrained_state_dict:
            self.model_pos.load_state_dict(
                self.pretrained_state_dict['model_pos'], strict=False)
        else:
            logging.error(
                'Not load model pos from pretrained_state_dict, not in pretrained_state_dict'
            )

        if 'model_traj' in self.pretrained_state_dict:
            self.model_traj.load_state_dict(
                self.pretrained_state_dict['model_traj'], strict=False)
        else:
            logging.error(
                'Not load model traj from pretrained_state_dict, not in pretrained_state_dict'
            )
        logging.info('Load pretrained model done.')

    def preprocess(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Proprocess of 2D input joints.

        Args:
            input (Dict[str, Any]): [NUM_FRAME, NUM_JOINTS, 2], input 2d human body keypoints.

        Returns:
            Dict[str, Any]: canonical 2d points and root relative joints.
        """
        if 'cuda' == input.device.type:
            input = input.data.cpu().numpy()
        elif 'cpu' == input.device.type:
            input = input.data.numpy()
        pose2d = input

        pose2d_canonical = self.canonicalize_2Ds(
            pose2d, self.cfg.model.INPUT.FOCAL_LENGTH,
            self.cfg.model.INPUT.CENTER)
        pose2d_normalized = self.normalize_screen_coordinates(
            pose2d, self.cfg.model.INPUT.RES_W, self.cfg.model.INPUT.RES_H)
        pose2d_rr = pose2d_normalized
        pose2d_rr[:, 1:] -= pose2d_rr[:, :1]

        # expand [NUM_FRAME, NUM_JOINTS, 2] to [1, NUM_FRAME, NUM_JOINTS, 2]
        pose2d_rr = np.expand_dims(
            np.pad(
                pose2d_rr,
                ((self.pad + self.causal_shift, self.pad - self.causal_shift),
                 (0, 0), (0, 0)), 'edge'),
            axis=0)
        pose2d_canonical = np.expand_dims(
            np.pad(
                pose2d_canonical,
                ((self.pad + self.causal_shift, self.pad - self.causal_shift),
                 (0, 0), (0, 0)), 'edge'),
            axis=0)
        pose2d_rr = torch.from_numpy(pose2d_rr.astype(np.float32))
        pose2d_canonical = torch.from_numpy(
            pose2d_canonical.astype(np.float32))

        inputs_2d = pose2d_rr.clone()
        if torch.cuda.is_available():
            inputs_2d = inputs_2d.cuda(non_blocking=True)

        # Positional model
        if self.cfg.model.MODEL.USE_2D_OFFSETS:
            inputs_2d[:, :, 0] = 0
        else:
            inputs_2d[:, :, 1:] += inputs_2d[:, :, :1]

        return {
            'inputs_2d': inputs_2d,
            'pose2d_rr': pose2d_rr,
            'pose2d_canonical': pose2d_canonical
        }

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """3D human pose estimation.

        Args:
            input (Dict):
                inputs_2d:  [1, NUM_FRAME, NUM_JOINTS, 2]
                pose2d_rr:  [1, NUM_FRAME, NUM_JOINTS, 2]
                pose2d_canonical: [1, NUM_FRAME, NUM_JOINTS, 2]
                NUM_FRAME = max(receptive_filed + video_frame_number, video_frame_number)

        Returns:
            Dict[str, Any]:
                "camera_pose": Tensor, [1, NUM_FRAME, OUT_NUM_JOINTS, OUT_3D_FEATURE_DIM],
                    3D human pose keypoints in camera frame.
                "camera_traj": Tensor, [1, NUM_FRAME, 1, 3],
                    root keypoints coordinates in camere frame.
        """
        inputs_2d = input['inputs_2d']
        pose2d_rr = input['pose2d_rr']
        pose2d_canonical = input['pose2d_canonical']
        with torch.no_grad():
            # predict 3D pose keypoints
            predicted_3d_pos = self.model_pos(inputs_2d)

            # predict global trajectory
            b1, w1, n1, d1 = inputs_2d.shape

            input_pose2d_abs = self.get_abs_2d_pts(w1, pose2d_rr,
                                                   pose2d_canonical)
            b1, w1, n1, d1 = input_pose2d_abs.size()
            b2, w2, n2, d2 = predicted_3d_pos.size()

            if torch.cuda.is_available():
                input_pose2d_abs = input_pose2d_abs.cuda(non_blocking=True)

            predicted_3d_traj = self.model_traj(
                input_pose2d_abs.view(b1, w1, n1 * d1),
                predicted_3d_pos.view(b2 * w2, n2 * d2)).view(b2, w2, -1, 3)

            predict_dict = {
                KeypointsTypes.POSES_CAMERA: predicted_3d_pos,
                KeypointsTypes.POSES_TRAJ: predicted_3d_traj
            }

        return predict_dict

    def get_abs_2d_pts(self, input_video_frame_num, pose2d_rr,
                       pose2d_canonical):
        pad = self.pad
        w = input_video_frame_num - pad * 2

        lst_pose2d_rr = []
        lst_pose2d_cannoical = []
        for i in range(pad, w + pad):
            lst_pose2d_rr.append(pose2d_rr[:, i - pad:i + pad + 1])
            lst_pose2d_cannoical.append(pose2d_canonical[:,
                                                         i - pad:i + pad + 1])

        input_pose2d_rr = torch.cat(lst_pose2d_cannoical, axis=0)
        input_pose2d_cannoical = torch.cat(lst_pose2d_cannoical, axis=0)

        if self.cfg.model.MODEL.USE_CANONICAL_COORDS:
            input_pose2d_abs = input_pose2d_cannoical.clone()
        else:
            input_pose2d_abs = input_pose2d_rr.clone()
            input_pose2d_abs[:, :, 1:] += input_pose2d_abs[:, :, :1]

        return input_pose2d_abs

    def canonicalize_2Ds(self, pos2d, f, c):
        cs = np.array([c[0], c[1]]).reshape(1, 1, 2)
        fs = np.array([f[0], f[1]]).reshape(1, 1, 2)
        canoical_2Ds = (pos2d - cs) / fs
        return canoical_2Ds

    def normalize_screen_coordinates(self, X, w, h):
        assert X.shape[-1] == 2

        # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
        return X / w * 2 - [1, h / w]
