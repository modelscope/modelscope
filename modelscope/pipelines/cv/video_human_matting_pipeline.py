# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict

import cv2
import numpy as np
import torch

from modelscope.metainfo import Pipelines
from modelscope.models.cv.video_human_matting import preprocess
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.video_human_matting, module_name=Pipelines.video_human_matting)
class VideoHumanMattingPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a video human matting pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        logger.info('load model done')

    def preprocess(self, input) -> Input:
        return input

    def forward(self, input: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        video_path = input['video_input_path']
        out_path = input['output_path']
        render = forward_params.get('render', False)
        video_input = cv2.VideoCapture(video_path)
        fps = video_input.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        success, frame = video_input.read()
        h, w = frame.shape[:2]
        scale = 512 / max(h, w)
        video_save = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        masks = []
        rec = [None] * 4
        self.model = self.model.to(self.device)
        logger.info('matting start using ' + self.device)
        with torch.no_grad():
            while True:
                if frame is None:
                    break
                frame_tensor = preprocess(frame)
                pha, *rec = self.model.model(
                    frame_tensor.to(self.device), *rec, downsample_ratio=scale)
                mask = pha * 255
                mask = mask[0].data.cpu().numpy().transpose(1, 2, 0)
                com = mask.repeat(3, 2).astype(np.uint8)
                video_save.write(com)
                masks.append((mask / 255).astype(np.uint8))
                success, frame = video_input.read()
        logger.info('matting process done')
        video_input.release()
        video_save.release()

        return {
            OutputKeys.MASKS: None if render else masks,
            OutputKeys.OUTPUT_VIDEO: out_path
        }

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
