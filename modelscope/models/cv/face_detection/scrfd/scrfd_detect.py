# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from copy import deepcopy
from typing import Any, Dict

import torch

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['ScrfdDetect']


@MODELS.register_module(Tasks.face_detection, module_name=Models.scrfd)
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
        cfg = Config.fromfile(osp.join(model_dir, 'mmcv_scrfd.py'))
        ckpt_path = osp.join(model_dir, ModelFile.TORCH_MODEL_BIN_FILE)
        cfg.model.test_cfg.score_thr = kwargs.get('score_thr', 0.3)
        detector = build_detector(cfg.model)
        logger.info(f'loading model from {ckpt_path}')
        device = torch.device(
            f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
        load_checkpoint(detector, ckpt_path, map_location=device)
        detector = MMDataParallel(detector, device_ids=[0])
        detector.eval()
        self.detector = detector
        logger.info('load model done')

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        result = self.detector(
            return_loss=False,
            rescale=True,
            img=[input['img'][0].unsqueeze(0)],
            img_metas=[[dict(input['img_metas'][0].data)]],
            output_results=2)
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

    def postprocess(self, input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return input
