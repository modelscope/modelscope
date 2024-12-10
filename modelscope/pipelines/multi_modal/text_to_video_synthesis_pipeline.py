# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import tempfile
from typing import Any, Dict, Optional

import cv2
import torch
import torchvision
from einops import rearrange

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.text_to_video_synthesis,
    module_name=Pipelines.text_to_video_synthesis)
class TextToVideoSynthesisPipeline(Pipeline):
    r""" Text To Video Synthesis Pipeline.

    Examples:
    >>> from modelscope.pipelines import pipeline
    >>> from modelscope.outputs import OutputKeys

    >>> p = pipeline('text-to-video-synthesis', 'damo/text-to-video-synthesis')
    >>> test_text = {
    >>>         'text': 'A panda eating bamboo on a rock.',
    >>>     }
    >>> p(test_text,)

    >>>  {OutputKeys.OUTPUT_VIDEO: path-to-the-generated-video}
    >>>
    """

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a kws pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)

    def preprocess(self, input: Input, **preprocess_params) -> Dict[str, Any]:
        self.model.clip_encoder.to(self.model.device)
        text_emb = self.model.clip_encoder(input['text'])
        text_emb_zero = self.model.clip_encoder('')
        if self.model.config.model.model_args.tiny_gpu == 1:
            self.model.clip_encoder.to('cpu')
        out_height = input['height'] if 'height' in input else 256
        out_width = input['width'] if 'height' in input else 256
        return {
            'text_emb': text_emb,
            'text_emb_zero': text_emb_zero,
            'out_height': out_height,
            'out_width': out_width
        }

    def forward(self, input: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        video = self.model(input)
        return {'video': video}

    def postprocess(self, inputs: Dict[str, Any],
                    **post_params) -> Dict[str, Any]:
        video = tensor2vid(inputs['video'])
        output_video_path = post_params.get('output_video', None)
        temp_video_file = False
        if output_video_path is None:
            output_video_path = tempfile.NamedTemporaryFile(suffix='.mp4').name
            temp_video_file = True

        # Ensure video is a list of frames with shape (h, w, c)
        frames = [torch.from_numpy(frame) for frame in video]
        # Stack frames along a new dimension to create a 4D tensor (T, H, W, C)
        imgs_tensor = torch.stack(frames, dim=0)

        torchvision.io.write_video(
            output_video_path,
            imgs_tensor,
            fps=8,
            video_codec='h264',
            options={'crf': '10'})
        if temp_video_file:
            video_file_content = b''
            with open(output_video_path, 'rb') as f:
                video_file_content = f.read()
            os.remove(output_video_path)
            return {OutputKeys.OUTPUT_VIDEO: video_file_content}
        else:
            return {OutputKeys.OUTPUT_VIDEO: output_video_path}


def tensor2vid(video, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    mean = torch.tensor(
        mean, device=video.device).reshape(1, -1, 1, 1, 1)  # ncfhw
    std = torch.tensor(
        std, device=video.device).reshape(1, -1, 1, 1, 1)  # ncfhw
    video = video.mul_(std).add_(mean)  # unnormalize back to [0,1]
    video.clamp_(0, 1)
    images = rearrange(video, 'i c f h w -> f h (i w) c')
    images = images.unbind(dim=0)
    images = [(image.numpy() * 255).astype('uint8')
              for image in images]  # f h w c
    return images
