# Copyright (c) Alibaba, Inc. and its affiliates.

import tempfile
from typing import Any, Dict, Optional

import cv2
import torch
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
        return {'text_emb': text_emb, 'text_emb_zero': text_emb_zero}

    def forward(self, input: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        video = self.model(input)
        return {'video': video}

    def postprocess(self, inputs: Dict[str, Any],
                    **post_params) -> Dict[str, Any]:
        video = tensor2vid(inputs['video'])
        output_video_path = post_params.get('output_video', None)
        if output_video_path is None:
            output_video_path = tempfile.NamedTemporaryFile(suffix='.mp4').name

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w, c = video[0].shape
        video_writer = cv2.VideoWriter(
            output_video_path, fourcc, fps=8, frameSize=(w, h))
        for i in range(len(video)):
            img = cv2.cvtColor(video[i], cv2.COLOR_RGB2BGR)
            video_writer.write(img)
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
