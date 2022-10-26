# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

import cv2
import numpy as np
import torch

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .one_stage_detector import OneStageDetector

logger = get_logger()


def load_model_weight(model_dir, device):
    checkpoint = torch.load(
        '{}/{}'.format(model_dir, ModelFile.TORCH_MODEL_BIN_FILE),
        map_location=device)
    state_dict = checkpoint['state_dict'].copy()
    for k in checkpoint['state_dict']:
        if k.startswith('avg_model.'):
            v = state_dict.pop(k)
            state_dict[k[4:]] = v

    return state_dict


@MODELS.register_module(
    Tasks.face_human_hand_detection,
    module_name=Models.face_human_hand_detection)
class NanoDetForFaceHumanHandDetection(TorchModel):

    def __init__(self, model_dir, device_id=0, *args, **kwargs):

        super().__init__(
            model_dir=model_dir, device_id=device_id, *args, **kwargs)

        self.model = OneStageDetector()
        if torch.cuda.is_available():
            self.device = 'cuda'
            logger.info('Use GPU ')
        else:
            self.device = 'cpu'
            logger.info('Use CPU')

        self.state_dict = load_model_weight(model_dir, self.device)
        self.model.load_state_dict(self.state_dict, strict=False)
        self.model.eval()
        self.model.to(self.device)

    def forward(self, x):
        pred_result = self.model.inference(x)
        return pred_result


def naive_collate(batch):
    elem = batch[0]
    if isinstance(elem, dict):
        return {key: naive_collate([d[key] for d in batch]) for key in elem}
    else:
        return batch


def get_resize_matrix(raw_shape, dst_shape):

    r_w, r_h = raw_shape
    d_w, d_h = dst_shape
    Rs = np.eye(3)

    Rs[0, 0] *= d_w / r_w
    Rs[1, 1] *= d_h / r_h
    return Rs


def color_aug_and_norm(meta, mean, std):
    img = meta['img'].astype(np.float32) / 255
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3) / 255
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3) / 255
    img = (img - mean) / std
    meta['img'] = img
    return meta


def img_process(meta, mean, std):
    raw_img = meta['img']
    height = raw_img.shape[0]
    width = raw_img.shape[1]
    dst_shape = [320, 320]
    M = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    ResizeM = get_resize_matrix((width, height), dst_shape)
    M = ResizeM @ M
    img = cv2.warpPerspective(raw_img, M, dsize=tuple(dst_shape))
    meta['img'] = img
    meta['warp_matrix'] = M
    meta = color_aug_and_norm(meta, mean, std)
    return meta


def overlay_bbox_cv(dets, class_names, score_thresh):
    all_box = []
    for label in dets:
        for bbox in dets[label]:
            score = bbox[-1]
            if score > score_thresh:
                x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                all_box.append([label, x0, y0, x1, y1, score])
    all_box.sort(key=lambda v: v[5])
    return all_box


mean = [103.53, 116.28, 123.675]
std = [57.375, 57.12, 58.395]
class_names = ['person', 'face', 'hand']


def inference(model, device, img):
    img = img.cpu().numpy()
    img_info = {'id': 0}
    height, width = img.shape[:2]
    img_info['height'] = height
    img_info['width'] = width
    meta = dict(img_info=img_info, raw_img=img, img=img)

    meta = img_process(meta, mean, std)
    meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1)).to(device)
    meta = naive_collate([meta])
    meta['img'] = (meta['img'][0]).reshape(1, 3, 320, 320)
    with torch.no_grad():
        res = model(meta)
    result = overlay_bbox_cv(res[0], class_names, score_thresh=0.35)
    cls_list, bbox_list, score_list = [], [], []
    for pred in result:
        cls_list.append(pred[0])
        bbox_list.append([pred[1], pred[2], pred[3], pred[4]])
        score_list.append(pred[5])
    return cls_list, bbox_list, score_list
