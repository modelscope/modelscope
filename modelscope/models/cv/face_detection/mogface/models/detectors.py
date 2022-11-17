# The implementation is based on MogFace, available at
# https://github.com/damo-cv/MogFace
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks
from .mogface import MogFace
from .utils import MogPriorBox, mogdecode, py_cpu_nms


@MODELS.register_module(Tasks.face_detection, module_name=Models.mogface)
class MogFaceDetector(TorchModel):

    def __init__(self, model_path, device='cuda'):
        super().__init__(model_path)
        cudnn.benchmark = True
        self.model_path = model_path
        self.device = device
        self.net = MogFace()
        self.load_model()
        self.net = self.net.to(device)

        self.mean = np.array([[104, 117, 123]])

    def load_model(self, load_to_cpu=False):
        pretrained_dict = torch.load(
            self.model_path, map_location=torch.device('cpu'))
        self.net.load_state_dict(pretrained_dict, strict=False)
        self.net.eval()

    def forward(self, input):
        img_raw = input['img']
        img = np.array(img_raw.cpu().detach())
        img = img[:, :, ::-1]

        im_height, im_width = img.shape[:2]
        ss = 1.0
        # tricky
        if max(im_height, im_width) > 1500:
            ss = 1000.0 / max(im_height, im_width)
            img = cv2.resize(img, (0, 0), fx=ss, fy=ss)
            im_height, im_width = img.shape[:2]

        scale = torch.Tensor(
            [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= np.array([[103.53, 116.28, 123.675]])
        img /= np.array([[57.375, 57.120003, 58.395]])
        img /= 255
        img = img[:, :, ::-1].copy()
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        conf, loc = self.net(img)  # forward pass

        confidence_threshold = 0.82
        nms_threshold = 0.4
        top_k = 5000
        keep_top_k = 750

        priorbox = MogPriorBox(scale_list=[0.68])
        priors = priorbox(im_height, im_width)
        priors = torch.tensor(priors).to(self.device)
        prior_data = priors.data

        boxes = mogdecode(loc.data.squeeze(0), prior_data)
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 0]

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]

        return dets / ss
