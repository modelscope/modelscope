# Copyright (c) Alibaba, Inc. and its affiliates.

import io
from typing import Any, Dict

import librosa
import numpy as np
import soundfile as sf
from sympy import true
import torch

from modelscope.fileio import File
from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.audio.audio_utils import audio_norm
from modelscope.utils.constant import Tasks
__all__ = ['VCPipeline']

@PIPELINES.register_module(
    Tasks.voice_conversion,
    module_name=Pipelines.voice_conversion)
class VCPipeline(Pipeline):
    r"""ANS (Acoustic Noise Suppression) Inference Pipeline .

    When invoke the class with pipeline.__call__(), it accept only one parameter:
        inputs(str): the path of wav file
    """
    SAMPLE_RATE = 16000

    def __init__(self, model, **kwargs):
        """
        use `model` and `preprocessor` to create a kws pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        self.model.eval()
        # self.stream_mode = kwargs.get('stream_mode', False)

    def preprocess(self, inputs: Input, **preprocess_params) -> Dict[str, Any]:
        # print(inputs)
        if  'source_wav' not in inputs:
            raise TypeError(f'source_wav not in inputs.')
        if  'target_wav' not in inputs:
            raise TypeError(f'target_wav not in inputs.')
        if  'save_path' not in inputs:
            raise TypeError(f'save_path not in inputs.')
        return inputs

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        self.model.convert(inputs['source_wav'],inputs['target_wav'],inputs['save_path'])
        return {"success":true}

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
       
        return inputs

