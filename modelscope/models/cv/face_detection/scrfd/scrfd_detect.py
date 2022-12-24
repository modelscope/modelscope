# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from copy import deepcopy
from typing import Any, Dict, List, Union

import numpy as np
import torch

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.outputs.cv_outputs import DetectionOutput
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['ScrfdDetect']


@MODELS.register_module(Tasks.face_detection, module_name=Models.scrfd)
@MODELS.register_module(Tasks.card_detection, module_name=Models.scrfd)
class ScrfdDetect(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the face detection model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)
        from mmcv import Config
        from mmcv.parallel import MMDataParallel
        from mmcv.runner import load_checkpoint
        from mmdet.models import build_detector
        from modelscope.models.cv.face_detection.scrfd.mmdet_patch.datasets import RetinaFaceDataset
        from modelscope.models.cv.face_detection.scrfd.mmdet_patch.datasets.pipelines import RandomSquareCrop
        from modelscope.models.cv.face_detection.scrfd.mmdet_patch.models.backbones import ResNetV1e
        from modelscope.models.cv.face_detection.scrfd.mmdet_patch.models.dense_heads import SCRFDHead
        from modelscope.models.cv.face_detection.scrfd.mmdet_patch.models.detectors import SCRFD
        cfg_file = kwargs.get('config_file', 'mmcv_scrfd.py')
        cfg = Config.fromfile(osp.join(model_dir, cfg_file))
        model_file = kwargs.get('model_file', ModelFile.TORCH_MODEL_BIN_FILE)
        ckpt_path = osp.join(model_dir, model_file)
        cfg.model.test_cfg.score_thr = kwargs.get('score_thr', 0.3)
        detector = build_detector(cfg.model)
        logger.info(f'loading model from {ckpt_path}')
        load_checkpoint(detector, ckpt_path, map_location='cpu')
        detector = MMDataParallel(detector, device_ids=[0])
        detector.eval()
        self.detector = detector
        logger.info('load model done')

    def forward(
        self, img: Union[torch.Tensor, List[torch.Tensor]],
        img_metas: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]
    ) -> DetectionOutput:
        """
        Args:
            img (`torch.Tensor` or `List[torch.Tensor]`): batched image tensor or list of
                batched image tensor, shape of each tensor is [N, h, w, 3]. When input is
                a list, each element is a different augmentation image to do multi-view
                augmentation test.
            img_metas (`List[List[Dict[str, Any]]]`): image meta info.

        Return:
            `:obj:DetectionOutput`
        """
        if isinstance(img, torch.Tensor):
            img = [img]
            img_metas = [img_metas]

        result = self.detector(
            return_loss=False,
            rescale=True,
            img=img,
            img_metas=img_metas,
            output_results=2)
        assert result is not None
        result = result[0][0]
        bboxes = result[:, :4]
        kpss = result[:, 5:]
        scores = result[:, 4]
        return DetectionOutput(scores=scores, boxes=bboxes, keypoints=kpss)

    def postprocess(self, detection_out: DetectionOutput,
                    **kwargs) -> Dict[str, Any]:
        scores = detection_out['scores'].tolist()
        boxes = detection_out['boxes'].tolist()
        kpss = detection_out['keypoints'].tolist()
        return {
            OutputKeys.SCORES: scores,
            OutputKeys.BOXES: boxes,
            OutputKeys.KEYPOINTS: kpss
        }
