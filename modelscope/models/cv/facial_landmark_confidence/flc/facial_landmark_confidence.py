# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from .manual_landmark_net import LandmarkConfidence


@MODELS.register_module(Tasks.face_2d_keypoints, module_name=Models.flc)
class FacialLandmarkConfidence(TorchModel):

    def __init__(self, model_path, device='cuda'):
        super().__init__(model_path)
        cudnn.benchmark = True
        self.model_path = model_path
        self.device = device
        self.cfg_path = model_path.replace(ModelFile.TORCH_MODEL_FILE,
                                           ModelFile.CONFIGURATION)
        self.landmark_count = 5
        self.net = LandmarkConfidence(landmark_count=self.landmark_count)
        self.load_model()
        self.net = self.net.to(device)

    def load_model(self, load_to_cpu=False):
        pretrained_dict = torch.load(
            self.model_path, map_location=torch.device('cpu'))['state_dict']
        pretrained_dict['rp_net.binary_cls.weight'] = 32.0 * F.normalize(
            pretrained_dict['rp_net.binary_cls.weight'], dim=1).t()
        self.net.load_state_dict(pretrained_dict, strict=True)
        self.net.eval()

    def forward(self, input):
        img_org = input['orig_img']
        bbox = input['bbox']
        img_org = img_org.cpu().numpy()

        image_height = img_org.shape[0]
        image_width = img_org.shape[1]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(image_width, int(bbox[2]))
        y2 = min(image_height, int(bbox[3]))
        box_w = x2 - x1 + 1
        box_h = y2 - y1 + 1
        if box_h > box_w:
            delta = box_h - box_w
            dy = edy = 0
            dx = delta // 2
            edx = delta - dx
        else:
            dx = edx = 0
            delta = box_w - box_h
            dy = delta // 2
            edy = delta - dy

        cv_img = img_org[y1:y2, x1:x2]
        if dx > 0 or dy > 0 or edx > 0 or edy > 0:
            cv_img = cv2.copyMakeBorder(cv_img, dy, edy, dx, edx,
                                        cv2.BORDER_CONSTANT, 0)
        inter_x = cv_img.shape[1]
        inter_y = cv_img.shape[0]

        cv_img = cv2.resize(cv_img, (120, 120))

        cv_img = cv_img.transpose((2, 0, 1))

        input_blob = torch.from_numpy(cv_img[np.newaxis, :, :, :].astype(
            np.float32))

        tmp_conf_lms, tmp_feat, tmp_conf_resp, tmp_nose = self.net(
            input_blob.to(self.device))
        conf_lms = tmp_conf_lms.cpu().numpy().squeeze()
        feat = tmp_feat.cpu().numpy().squeeze()

        pts5pt = []
        for i in range(feat.shape[0]):
            if i < self.landmark_count:
                pts5pt.append(feat[i] * inter_x - dx + x1)
            else:
                pts5pt.append(feat[i] * inter_y - dy + y1)

        lm5pt = np.array(pts5pt).reshape(2, 5).T
        return lm5pt, conf_lms
