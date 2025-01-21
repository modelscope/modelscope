# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict

import cv2
import numpy as np
import torch

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .model import FlowGenerator
from .person_info import PersonInfo
from .pose_estimator.body import Body
from .slim_utils import image_warp_grid1, resize_on_long_side

logger = get_logger()

__all__ = ['ImageBodyReshaping']


@MODELS.register_module(
    Tasks.image_body_reshaping, module_name=Models.image_body_reshaping)
class ImageBodyReshaping(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the image body reshaping model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.degree = 1.0
        self.reshape_model = FlowGenerator(n_channels=16).to(self.device)
        model_path = os.path.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        checkpoints = torch.load(model_path, map_location=torch.device('cpu'))
        self.reshape_model.load_state_dict(
            checkpoints['state_dict'], strict=True)
        self.reshape_model.eval()
        logger.info('load body reshaping model done')

        pose_model_ckpt = os.path.join(model_dir, 'body_pose_model.pth')
        self.pose_esti = Body(pose_model_ckpt, self.device)
        logger.info('load pose model done')

    def pred_joints(self, img):
        if img is None:
            return None
        small_src, resize_scale = resize_on_long_side(img, 300)
        body_joints = self.pose_esti(small_src)

        if body_joints.shape[0] >= 1:
            body_joints[:, :, :2] = body_joints[:, :, :2] / resize_scale

        return body_joints

    def pred_flow(self, img):

        body_joints = self.pred_joints(img)
        small_size = 1200

        if img.shape[0] > small_size or img.shape[1] > small_size:
            _img, _scale = resize_on_long_side(img, small_size)
            body_joints[:, :, :2] = body_joints[:, :, :2] * _scale
        else:
            _img = img

        # We only reshape one person
        if body_joints.shape[0] < 1 or body_joints.shape[0] > 1:
            return None

        person = PersonInfo(body_joints[0])

        with torch.no_grad():
            person_pred = person.pred_flow(_img, self.reshape_model,
                                           self.device)

        flow = np.dstack((person_pred['rDx'], person_pred['rDy']))

        scale = img.shape[0] * 1.0 / flow.shape[0]

        flow = cv2.resize(flow, (img.shape[1], img.shape[0]))
        flow *= scale

        return flow

    def warp(self, src_img, flow):

        X_flow = flow[..., 0]
        Y_flow = flow[..., 1]

        X_flow = np.ascontiguousarray(X_flow)
        Y_flow = np.ascontiguousarray(Y_flow)

        pred = image_warp_grid1(X_flow, Y_flow, src_img, 1.0, 0, 0)
        return pred

    def inference(self, img):
        img = img.cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        flow = self.pred_flow(img)

        if flow is None:
            return img

        assert flow.shape[:2] == img.shape[:2]

        mag, ang = cv2.cartToPolar(flow[..., 0] + 1e-8, flow[..., 1] + 1e-8)
        mag -= 3
        mag[mag <= 0] = 0

        x, y = cv2.polarToCart(mag, ang, angleInDegrees=False)
        flow = np.dstack((x, y))

        flow *= self.degree
        pred = self.warp(img, flow)
        out_img = np.clip(pred, 0, 255)
        logger.info('model inference done')

        return out_img.astype(np.uint8)
