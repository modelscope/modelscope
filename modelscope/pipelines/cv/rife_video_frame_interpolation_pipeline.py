# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import math
import os
import os.path as osp
import subprocess
import tempfile
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from modelscope.metainfo import Pipelines
from modelscope.models.cv.video_frame_interpolation.rife import RIFEModel
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.preprocessors.cv import VideoReader
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.video_frame_interpolation,
    module_name=Pipelines.rife_video_frame_interpolation)
class RIFEVideoFrameInterpolationPipeline(Pipeline):
    r""" RIFE Video Frame Interpolation Pipeline.

    Examples:

    >>> from modelscope.pipelines import pipeline
    >>> from modelscope.utils.constant import Tasks
    >>> from modelscope.outputs import OutputKeys

    >>> video = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/video_frame_interpolation_test.mp4'
    >>> video_frame_interpolation_pipeline = pipeline(Tasks.video_frame_interpolation,
    'Damo_XR_Lab/cv_rife_video-frame-interpolation')
    >>> result = video_frame_interpolation_pipeline(video)[OutputKeys.OUTPUT_VIDEO]
    >>> print('pipeline: the output video path is {}'.format(result))

    """

    def __init__(self,
                 model: Union[RIFEModel, str],
                 preprocessor=None,
                 **kwargs):
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        if (isinstance(model, str)):
            self.model = RIFEModel(model)
        logger.info('load video frame-interpolation done')

    def preprocess(self, input: Input, out_fps: float = 0) -> Dict[str, Any]:
        # Determine the input type
        if isinstance(input, str):
            video_reader = VideoReader(input)
        elif isinstance(input, dict):
            video_reader = VideoReader(input['video'])
        inputs = []
        for frame in video_reader:
            inputs.append(frame)
        fps = video_reader.fps

        for i, img in enumerate(inputs):
            img = torch.from_numpy(img.copy()).permute(2, 0, 1).float()
            inputs[i] = img.unsqueeze(0).to(self.model.device)

        out_fps = 2 * fps
        return {'video': inputs, 'fps': fps, 'out_fps': out_fps}

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        inputs = input['video']
        # fps = input['fps']
        out_fps = input['out_fps']
        video_len = len(inputs)

        h, w = inputs[0].shape[-2:]
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)

        outputs = []
        for i in range(video_len):
            if i == 0:
                outputs.append(inputs[i])
            elif i == video_len - 1:
                outputs.append(inputs[i])
            else:
                i0 = F.pad(inputs[i - 1] / 255., padding).to(self.model.device)
                i1 = F.pad(inputs[i] / 255., padding).to(self.model.device)
                output = self.model.inference(i0, i1)[:, :, :h, :w]
                output = output.cpu() * 255
                torch.cuda.empty_cache()
                outputs.append(output)
                outputs.append(inputs[i])
        return {'output': outputs, 'fps': out_fps}

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        output_video_path = kwargs.get('output_video', None)
        demo_service = kwargs.get('demo_service', True)
        if output_video_path is None:
            output_video_path = tempfile.NamedTemporaryFile(suffix='.mp4').name
        h, w = inputs['output'][0].shape[-2:]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc,
                                       inputs['fps'], (w, h))
        for i in range(len(inputs['output'])):
            img = inputs['output'][i]
            img = img[0].permute(1, 2, 0).byte().cpu().numpy()
            video_writer.write(img.astype(np.uint8))

        video_writer.release()
        if demo_service:
            assert os.system(
                'ffmpeg -version') == 0, 'ffmpeg is not installed correctly!'
            output_video_path_for_web = output_video_path[:-4] + '_web.mp4'
            convert_cmd = f'ffmpeg -i {output_video_path} -vcodec h264 -crf 5 {output_video_path_for_web}'
            subprocess.call(convert_cmd, shell=True)
            return {OutputKeys.OUTPUT_VIDEO: output_video_path_for_web}
        else:
            return {OutputKeys.OUTPUT_VIDEO: output_video_path}
