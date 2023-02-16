# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict

import cv2
import numpy as np
import torch
from moviepy.editor import ImageSequenceClip, VideoFileClip

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
        """ Video Human Matting Pipeline.
        use `model` to create a video human matting pipeline for prediction

        Example:

        >>> from modelscope.pipelines import pipeline
        >>> from modelscope.outputs import OutputKeys
        >>> from modelscope.utils.constant import Tasks
        >>> video_matting = pipeline(Tasks.video_human_matting, model='damo/cv_effnetv2_video-human-matting')
        >>> result_status = video_matting({
        'video_input_path':'https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/video_matting_test.mp4',
        'output_path':'matting_out.mp4'})
        >>> masks = result_status[OutputKeys.MASKS]
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
        if 'output_path' in input:
            out_path = input['output_path']
        else:
            out_path = 'output.mp4'
        video_input = cv2.VideoCapture(video_path)
        fps = video_input.get(cv2.CAP_PROP_FPS)
        success, frame = video_input.read()
        h, w = frame.shape[:2]
        scale = 512 / max(h, w)
        self.fps = fps
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
                mask = pha[0].data.cpu().numpy().transpose(1, 2, 0)
                masks.append(mask)
                success, frame = video_input.read()
        logger.info('matting process done')
        video_input.release()

        return {OutputKeys.MASKS: masks, OutputKeys.OUTPUT_VIDEO: out_path}

    def postprocess(self, inputs, **kwargs) -> Dict[str, Any]:
        render = kwargs.get('render', False)
        masks = inputs[OutputKeys.MASKS]
        output_path = inputs[OutputKeys.OUTPUT_VIDEO]
        frame_lst = []
        for mask in masks:
            com = (mask * 255).repeat(3, 2).astype(np.uint8)
            frame_lst.append(com)
        video = ImageSequenceClip(sequence=frame_lst, fps=self.fps)
        video.write_videofile(output_path, fps=self.fps, audio=False)
        del frame_lst

        result = {
            OutputKeys.MASKS: None if render else masks,
            OutputKeys.OUTPUT_VIDEO: output_path
        }
        return result
