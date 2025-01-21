# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp

import numpy as np
import torch

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from .mmdet_ms.backbones import ViT
from .mmdet_ms.dense_heads import RPNNHead
from .mmdet_ms.necks import FPNF
from .mmdet_ms.roi_heads import FCNMaskNHead, Shared4Conv1FCBBoxNHead


@MODELS.register_module(Tasks.human_detection, module_name=Models.detection)
@MODELS.register_module(
    Tasks.image_object_detection, module_name=Models.detection)
class DetectionModel(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """str -- model file root."""
        super().__init__(model_dir, *args, **kwargs)

        from mmcv.runner import load_checkpoint
        from mmdet.datasets import replace_ImageToTensor
        from mmdet.datasets.pipelines import Compose
        from mmdet.models import build_detector

        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        config_path = osp.join(model_dir, 'mmcv_config.py')
        config = Config.from_file(config_path)
        config.model.pretrained = None
        self.model = build_detector(config.model)

        checkpoint = load_checkpoint(
            self.model, model_path, map_location='cpu')
        self.class_names = checkpoint['meta']['CLASSES']
        config.test_pipeline[0].type = 'LoadImageFromWebcam'
        self.transform_input = Compose(
            replace_ImageToTensor(config.test_pipeline))
        self.model.cfg = config
        self.model.eval()
        self.score_thr = config.score_thr

    def inference(self, data):
        """data is dict,contain img and img_metas,follow with mmdet."""

        with torch.no_grad():
            results = self.model(return_loss=False, rescale=True, **data)
        return results

    def preprocess(self, image):
        """image is numpy return is dict contain img and img_metas,follow with mmdet."""

        from mmcv.parallel import collate, scatter
        data = dict(img=image)
        data = self.transform_input(data)
        data = collate([data], samples_per_gpu=1)
        data['img_metas'] = [
            img_metas.data[0] for img_metas in data['img_metas']
        ]
        data['img'] = [img.data[0] for img in data['img']]

        if next(self.model.parameters()).is_cuda:
            data = scatter(data, [next(self.model.parameters()).device])[0]

        return data

    def postprocess(self, inputs):

        if isinstance(inputs[0], tuple):
            bbox_result, _ = inputs[0]
        else:
            bbox_result, _ = inputs[0], None
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        bbox_result = np.vstack(bbox_result)
        scores = bbox_result[:, -1]
        inds = scores > self.score_thr
        if np.sum(np.array(inds).astype('int')) == 0:
            return None, None, None
        bboxes = bbox_result[inds, :]
        labels = labels[inds]
        scores = np.around(bboxes[:, 4], 6)
        bboxes = (bboxes[:, 0:4]).astype(int)
        labels = [self.class_names[i_label] for i_label in labels]
        return bboxes, scores, labels
