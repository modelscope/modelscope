# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import logging as logger
import os
import os.path as osp
import time

import cv2
import json
import numpy as np
import torch
from tqdm import tqdm

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.stream_yolo.data.data_augment import ValTransform
from modelscope.models.cv.stream_yolo.utils import (postprocess,
                                                    timestamp_format)
from modelscope.preprocessors import LoadImage
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from .exp.longshortnet_base import LongShortNetExp


@MODELS.register_module(
    group_key=Tasks.video_object_detection, module_name=Models.longshortnet)
class LongShortNet(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.depth = kwargs.get('depth', 0.33)
        self.width = kwargs.get('width', 0.50)
        self.num_classes = kwargs.get('num_classes', 8)
        self.test_size = kwargs.get('test_size', (960, 600))
        self.test_conf = kwargs.get('test_conf', 0.3)
        self.nmsthre = kwargs.get('nmsthre', 0.55)
        self.label_mapping = kwargs.get('labels', [
            'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
            'traffic light', 'stop sign'
        ])
        self.model_name = kwargs.get('model_name', 'longshortnet_s.pt')
        self.short_cfg = kwargs.get(
            'short_cfg',
            dict(
                frame_num=1,
                delta=1,
                with_short_cut=False,
                out_channels=[
                    ((64, 128, 256), 1),
                ],
            ))
        self.long_cfg = kwargs.get(
            'long_cfg',
            dict(
                frame_num=3,
                delta=1,
                with_short_cut=False,
                include_current_frame=False,
                out_channels=[
                    ((21, 42, 85), 3),
                ],
            ))
        self.merge_cfg = kwargs.get(
            'merge_cfg', dict(
                merge_form='long_fusion',
                with_short_cut=True,
            ))

        self.exp = LongShortNetExp()

        self.exp.depth = self.depth
        self.exp.width = self.width
        self.exp.num_classes = self.num_classes
        self.exp.test_size = self.test_size
        self.exp.test_conf = self.test_conf
        self.exp.nmsthre = self.nmsthre
        self.exp.short_cfg = self.short_cfg
        self.exp.long_cfg = self.long_cfg
        self.exp.merge_cfg = self.merge_cfg

        # build model
        self.model = self.exp.get_model()
        model_path = osp.join(model_dir, self.model_name)
        ckpt = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(ckpt['model'])
        self.preproc = ValTransform(legacy=False)

    def forward(self, inputs):
        return self.inference_video(inputs)

    def postprocess(self, input):
        outputs = postprocess(
            input,
            self.num_classes,
            self.test_conf,
            self.nmsthre,
            class_agnostic=True)

        if len(outputs) == 1 and (outputs[0] is not None):
            bboxes = outputs[0][:, 0:4].cpu().numpy() / self.resize_ratio
            scores = outputs[0][:, 5].cpu().numpy()
            labels = outputs[0][:, 6].cpu().int().numpy()
            pred_label_names = []
            for lab in labels:
                pred_label_names.append(self.label_mapping[lab])
        else:
            bboxes = np.asarray([])
            scores = np.asarray([])
            pred_label_names = np.asarray([])

        return bboxes, scores, pred_label_names

    def inference_video(self, v_path):
        outputs = []
        capture = cv2.VideoCapture(v_path)
        self.fps = capture.get(cv2.CAP_PROP_FPS)
        self.ori_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.ori_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.ori_size = (self.ori_width, self.ori_height)
        self.resize_ratio = min(self.test_size[0] / self.ori_size[0],
                                self.test_size[1] / self.ori_size[1])
        self.device = next(self.model.parameters()).device
        frame_idx = 0

        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break
            if frame_idx == 0:
                short_imgs_queue = [
                    frame.copy() for _ in range(self.short_cfg['frame_num'])
                ]
                long_imgs_queue = [
                    frame.copy() for _ in range(self.long_cfg['frame_num'])
                ]
                short_imgs_queue = [
                    cv2.resize(
                        x, self.test_size,
                        interpolation=cv2.INTER_LINEAR).astype(np.uint8)
                    for x in short_imgs_queue
                ]
                long_imgs_queue = [
                    cv2.resize(
                        x, self.test_size,
                        interpolation=cv2.INTER_LINEAR).astype(np.uint8)
                    for x in long_imgs_queue
                ]
                short_imgs_queue = [
                    self.preproc(x, None,
                                 (self.test_size[1], self.test_size[0]))[0]
                    for x in short_imgs_queue
                ]
                long_imgs_queue = [
                    self.preproc(x, None,
                                 (self.test_size[1], self.test_size[0]))[0]
                    for x in long_imgs_queue
                ]
            else:
                long_imgs_queue = long_imgs_queue[1:] + short_imgs_queue[:]
                short_imgs_queue = [
                    frame.copy() for _ in range(self.short_cfg['frame_num'])
                ]
                short_imgs_queue = [
                    cv2.resize(
                        x, self.test_size,
                        interpolation=cv2.INTER_LINEAR).astype(np.uint8)
                    for x in short_imgs_queue
                ]
                short_imgs_queue = [
                    self.preproc(x, None,
                                 (self.test_size[1], self.test_size[0]))[0]
                    for x in short_imgs_queue
                ]

            short_img = np.concatenate(short_imgs_queue, axis=0)
            long_img = np.concatenate(long_imgs_queue, axis=0)
            short_img = torch.from_numpy(short_img).unsqueeze(0)
            long_img = torch.from_numpy(long_img).unsqueeze(0)

            short_img = short_img.to(self.device)
            long_img = long_img.to(self.device)

            output = self.model((short_img, long_img))
            output = self.postprocess(output)

            output += (timestamp_format(seconds=frame_idx / self.fps), )

            outputs.append(output)

            frame_idx += 1

        return outputs
