# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.body_3d_keypoints.hdformer.hdformer import HDFormer
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger


class KeypointsTypes(object):
    POSES_CAMERA = 'poses_camera'


logger = get_logger()


@MODELS.register_module(
    Tasks.body_3d_keypoints, module_name=Models.body_3d_keypoints_hdformer)
class HDFormerDetector(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.model_dir = model_dir

        cudnn.benchmark = True
        self.model_path = osp.join(self.model_dir, ModelFile.TORCH_MODEL_FILE)
        self.mean_std_2d = np.load(
            osp.join(self.model_dir, 'mean_std_2d.npy'), allow_pickle=True)
        self.mean_std_3d = np.load(
            osp.join(self.model_dir, 'mean_std_3d.npy'), allow_pickle=True)
        self.left_right_symmetry_2d = np.array(
            [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13])
        cfg_path = osp.join(self.model_dir, ModelFile.CONFIGURATION)
        self.cfg = Config.from_file(cfg_path)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.net = HDFormer(self.cfg.model.MODEL)

        self.load_model()
        self.net = self.net.to(self.device)

    def load_model(self, load_to_cpu=False):
        pretrained_dict = torch.load(
            self.model_path,
            map_location=torch.device('cuda')
            if torch.cuda.is_available() else torch.device('cpu'))
        self.net.load_state_dict(pretrained_dict['state_dict'], strict=False)
        self.net.eval()

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
        num_frames, num_joints, in_channels = pose2d.shape
        logger.info(f'2d pose frame number: {num_frames}')

        # [NUM_FRAME, NUM_JOINTS, 2]
        c = np.array(self.cfg.model.INPUT.center)
        f = np.array(self.cfg.model.INPUT.focal_length)
        self.window_size = self.cfg.model.INPUT.window_size
        receptive_field = self.cfg.model.INPUT.n_frames

        # split the 2D pose sequences into fixed length frames
        inputs_2d = []
        inputs_2d_flip = []
        n = 0
        indices = []
        while n + receptive_field <= num_frames:
            indices.append((n, n + receptive_field))
            n += self.window_size
        self.valid_length = n - self.window_size + receptive_field

        if 0 == len(indices):
            logger.warn(
                f'Fail to construct test sequences, total_frames = {num_frames}, \
                while receptive_filed ={receptive_field}')

        self.mean_2d = self.mean_std_2d[0]
        self.std_2d = self.mean_std_2d[1]
        for (start, end) in indices:
            data_2d = pose2d[start:end]
            data_2d = (data_2d - 0.5 - c) / f
            data_2d_flip = data_2d.copy()
            data_2d_flip[:, :, 0] *= -1
            data_2d_flip = data_2d_flip[:, self.left_right_symmetry_2d, :]
            data_2d_flip = (data_2d_flip - self.mean_2d) / self.std_2d

            data_2d = (data_2d - self.mean_2d) / self.std_2d
            data_2d = torch.from_numpy(data_2d.transpose(
                (2, 0, 1))).float()  # [C,T,V]

            data_2d_flip = torch.from_numpy(data_2d_flip.transpose(
                (2, 0, 1))).float()  # [C,T,V]

            inputs_2d.append(data_2d)
            inputs_2d_flip.append(data_2d_flip)

        self.mean_3d = self.mean_std_3d[0]
        self.std_3d = self.mean_std_3d[1]
        mean_3d = torch.from_numpy(self.mean_3d).float().unsqueeze(-1)
        mean_3d = mean_3d.permute(1, 2, 0)  # [3, 1, 17]
        std_3d = torch.from_numpy(self.std_3d).float().unsqueeze(-1)
        std_3d = std_3d.permute(1, 2, 0)

        return {
            'inputs_2d': inputs_2d,
            'inputs_2d_flip': inputs_2d_flip,
            'mean_3d': mean_3d,
            'std_3d': std_3d
        }

    def avg_flip(self, pre, pre_flip):
        left_right_symmetry = [
            0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13
        ]
        pre_flip[:, 0, :, :] *= -1
        pre_flip = pre_flip[:, :, :, left_right_symmetry]
        pred_avg = (pre + pre_flip) / 2.
        return pred_avg

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """3D human pose estimation.

        Args:
            input (Dict):
                inputs_2d:  [1, NUM_FRAME, NUM_JOINTS, 2]

        Returns:
            Dict[str, Any]:
                "camera_pose": Tensor, [1, NUM_FRAME, OUT_NUM_JOINTS, OUT_3D_FEATURE_DIM],
                    3D human pose keypoints in camera frame.
                "success": 3D pose estimation success or failed.
        """
        inputs_2d = input['inputs_2d']
        inputs_2d_flip = input['inputs_2d_flip']
        mean_3d = input['mean_3d']
        std_3d = input['std_3d']
        preds_3d = None
        vertex_pre = None

        if [] == inputs_2d:
            predict_dict = {'success': False, KeypointsTypes.POSES_CAMERA: []}
            return predict_dict

        with torch.no_grad():
            for i, pose_2d in enumerate(inputs_2d):
                pose_2d = pose_2d.unsqueeze(0).cuda(non_blocking=True) \
                    if torch.cuda.is_available() else pose_2d.unsqueeze(0)
                pose_2d_flip = inputs_2d_flip[i]
                pose_2d_flip = pose_2d_flip.unsqueeze(0).cuda(non_blocking=True) \
                    if torch.cuda.is_available() else pose_2d_flip.unsqueeze(0)
                mean_3d = mean_3d.unsqueeze(0).cuda(non_blocking=True) \
                    if torch.cuda.is_available() else mean_3d.unsqueeze(0)
                std_3d = std_3d.unsqueeze(0).cuda(non_blocking=True) \
                    if torch.cuda.is_available() else std_3d.unsqueeze(0)

                vertex_pre = self.net(pose_2d, mean_3d, std_3d)
                vertex_pre_flip = self.net(pose_2d_flip, mean_3d, std_3d)
                vertex_pre = self.avg_flip(vertex_pre, vertex_pre_flip)

                # concat the prediction results for each window_size
                predict_3d = vertex_pre.permute(
                    0, 2, 3, 1).contiguous()[0][:self.window_size]
                if preds_3d is None:
                    preds_3d = predict_3d
                else:
                    preds_3d = torch.concat((preds_3d, predict_3d), dim=0)
            remain_pose_results = vertex_pre.permute(
                0, 2, 3, 1).contiguous()[0][self.window_size:]
            preds_3d = torch.concat((preds_3d, remain_pose_results), dim=0)

        preds_3d = preds_3d.unsqueeze(0)  # add batch dim
        preds_3d = preds_3d / self.cfg.model.INPUT.res_w  # Normalize to [-1, 1]
        predict_dict = {'success': True, KeypointsTypes.POSES_CAMERA: preds_3d}

        return predict_dict
