# The GPEN implementation is also open-sourced by the authors,
# and available at https://github.com/yangxy/GPEN/blob/main/face_detect/retinaface_detection.py
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from .models.retinaface import RetinaFace
from .utils import PriorBox, decode, decode_landm, py_cpu_nms

cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'pretrain': False,
    'return_layers': {
        'layer2': 1,
        'layer3': 2,
        'layer4': 3
    },
    'in_channel': 256,
    'out_channel': 256
}


class RetinaFaceDetection(object):

    def __init__(self, model_path, device='cuda'):
        cudnn.benchmark = True
        self.model_path = model_path
        self.device = device
        self.cfg = cfg_re50
        self.net = RetinaFace(cfg=self.cfg)
        self.load_model()
        self.net = self.net.to(device)

        self.mean = torch.tensor([[[[104]], [[117]], [[123]]]]).to(device)

    def check_keys(self, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(self.net.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        assert len(
            used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    def remove_prefix(self, state_dict, prefix):
        new_state_dict = dict()
        # remove unnecessary 'module.'
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_state_dict[k[len(prefix):]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    def load_model(self, load_to_cpu=False):
        pretrained_dict = torch.load(
            self.model_path, map_location=torch.device('cpu'))
        if 'state_dict' in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'],
                                                 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(pretrained_dict)
        self.net.load_state_dict(pretrained_dict, strict=False)
        self.net.eval()

    def detect(self,
               img_raw,
               resize=1,
               confidence_threshold=0.9,
               nms_threshold=0.4,
               top_k=5000,
               keep_top_k=750,
               save_image=False):
        img = np.float32(img_raw)

        im_height, im_width = img.shape[:2]
        ss = 1.0
        # tricky
        if max(im_height, im_width) > 1500:
            ss = 1000.0 / max(im_height, im_width)
            img = cv2.resize(img, (0, 0), fx=ss, fy=ss)
            im_height, im_width = img.shape[:2]

        scale = torch.Tensor(
            [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        loc, conf, landms = self.net(img)  # forward pass
        del img

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(
            landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([
            im_width, im_height, im_width, im_height, im_width, im_height,
            im_width, im_height, im_width, im_height
        ])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]

        landms = landms.reshape((-1, 5, 2))
        landms = landms.transpose((0, 2, 1))
        landms = landms.reshape(
            -1,
            10,
        )
        return dets / ss, landms / ss

    def detect_tensor(self,
                      img,
                      resize=1,
                      confidence_threshold=0.9,
                      nms_threshold=0.4,
                      top_k=5000,
                      keep_top_k=750,
                      save_image=False):
        im_height, im_width = img.shape[-2:]
        ss = 1000 / max(im_height, im_width)
        img = F.interpolate(img, scale_factor=ss)
        im_height, im_width = img.shape[-2:]
        scale = torch.Tensor([im_width, im_height, im_width,
                              im_height]).to(self.device)
        img -= self.mean

        loc, conf, landms = self.net(img)  # forward pass

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(
            landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([
            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
            img.shape[3], img.shape[2]
        ])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]

        landms = landms.reshape((-1, 5, 2))
        landms = landms.transpose((0, 2, 1))
        landms = landms.reshape(
            -1,
            10,
        )
        return dets / ss, landms / ss
