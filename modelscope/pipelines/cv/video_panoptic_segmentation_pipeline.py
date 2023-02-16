# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import os.path as osp
from typing import Any, Dict

import cv2
import mmcv
import numpy as np
import torch
from tqdm import tqdm

from modelscope.metainfo import Pipelines
from modelscope.models.cv.video_panoptic_segmentation.video_k_net import \
    VideoKNet
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.video_panoptic_segmentation,
    module_name=Pipelines.video_panoptic_segmentation)
class VideoPanopticSegmentationPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a video panoptic segmentation pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, auto_collate=False, **kwargs)
        logger.info(f'loading model from {model}')
        model_path = osp.join(model, ModelFile.TORCH_MODEL_FILE)
        config_path = osp.join(model, ModelFile.CONFIGURATION)
        logger.info(f'loading config from {config_path}')
        self.cfg = Config.from_file(config_path)
        self.max_video_frames = kwargs.get('max_video_frames', 1000)

        self.model = VideoKNet(model)
        checkpoint = torch.load(
            model_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device).eval()
        logger.info('load model done')

        self.pad_size_divisor = 32
        self.mean = np.array([123.675, 116.28, 103.53], np.float32)
        self.std = np.array([58.395, 57.12, 57.375], np.float32)
        self.to_rgb = False

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if not isinstance(input, str):
            raise TypeError(f'input should be a str,'
                            f'  but got {type(input)}')
        frames = []
        img_metas = []
        iids = []
        cap = cv2.VideoCapture(input)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_idx = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx > self.max_video_frames:
                break

            norm_frame = mmcv.imnormalize(frame, self.mean, self.std,
                                          self.to_rgb)
            pad_frame = mmcv.impad_to_multiple(
                norm_frame, self.pad_size_divisor, pad_val=0)

            img_meta = {}
            img_meta['ori_shape'] = frame.shape
            img_meta['img_shape'] = frame.shape
            img_meta['pad_shape'] = pad_frame.shape
            img_meta['batch_input_shape'] = pad_frame.shape[0:2]
            img_meta['scale_factor'] = 1.0,
            img_meta['flip'] = False
            img_meta['flip_direction'] = None

            frames.append(pad_frame)
            img_metas.append([img_meta])
            iids.append(frame_idx)

            frame_idx += 1

        result = {
            'video_name': input,
            'imgs': np.array(frames),
            'img_metas': img_metas,
            'iids': iids,
        }
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        scores = []
        labels = []
        masks = []
        boxes = []
        track_ids = []
        for ii in tqdm(range(len(input['iids']))):
            img = input['imgs'][ii]
            img_meta = input['img_metas'][ii]
            iid = input['iids'][ii]

            x = np.transpose(img, [2, 0, 1])
            x = np.expand_dims(x, 0)
            x = torch.from_numpy(x).to(self.device)
            with torch.no_grad():
                segm_results = self.model(x, img_meta, rescale=True, iid=iid)

            _, _, _, vis_sem, vis_tracker, label, binary_mask, track_id, thing_bbox_for_tracking = segm_results
            scores.append([0.99] * len(label))
            labels.append(label)
            masks.append(binary_mask)
            boxes.append(thing_bbox_for_tracking)
            track_ids.append(track_id)

        output = {
            'scores': scores,
            'labels': labels,
            'masks': masks,
            'boxes': boxes,
            'uuid': track_ids
        }
        return output

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
