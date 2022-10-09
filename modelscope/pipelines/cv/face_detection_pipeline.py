# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict

import cv2
import numpy as np
import PIL
import torch

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.face_detection, module_name=Pipelines.face_detection)
class FaceDetectionPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a face detection pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        from mmcv import Config
        from mmcv.parallel import MMDataParallel
        from mmcv.runner import load_checkpoint
        from mmdet.models import build_detector
        from modelscope.models.cv.face_detection.mmdet_patch.datasets import RetinaFaceDataset
        from modelscope.models.cv.face_detection.mmdet_patch.datasets.pipelines import RandomSquareCrop
        from modelscope.models.cv.face_detection.mmdet_patch.models.backbones import ResNetV1e
        from modelscope.models.cv.face_detection.mmdet_patch.models.dense_heads import SCRFDHead
        from modelscope.models.cv.face_detection.mmdet_patch.models.detectors import SCRFD
        cfg = Config.fromfile(osp.join(model, 'mmcv_scrfd_10g_bnkps.py'))
        detector = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        ckpt_path = osp.join(model, ModelFile.TORCH_MODEL_BIN_FILE)
        logger.info(f'loading model from {ckpt_path}')
        device = torch.device(
            f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
        load_checkpoint(detector, ckpt_path, map_location=device)
        detector = MMDataParallel(detector, device_ids=[0])
        detector.eval()
        self.detector = detector
        logger.info('load model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_ndarray(input)
        img = img.astype(np.float32)
        pre_pipeline = [
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[128.0, 128.0, 128.0],
                        to_rgb=False),
                    dict(type='Pad', size=(640, 640), pad_val=0),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]
        from mmdet.datasets.pipelines import Compose
        pipeline = Compose(pre_pipeline)
        result = {}
        result['filename'] = ''
        result['ori_filename'] = ''
        result['img'] = img
        result['img_shape'] = img.shape
        result['ori_shape'] = img.shape
        result['img_fields'] = ['img']
        result = pipeline(result)
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:

        result = self.detector(
            return_loss=False,
            rescale=True,
            img=[input['img'][0].unsqueeze(0)],
            img_metas=[[dict(input['img_metas'][0].data)]])
        assert result is not None
        result = result[0][0]
        bboxes = result[:, :4].tolist()
        kpss = result[:, 5:].tolist()
        scores = result[:, 4].tolist()
        return {
            OutputKeys.SCORES: scores,
            OutputKeys.BOXES: bboxes,
            OutputKeys.KEYPOINTS: kpss
        }

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
