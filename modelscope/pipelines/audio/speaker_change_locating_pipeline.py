# Copyright (c) Alibaba, Inc. and its affiliates.

import io
from typing import Any, Dict, List, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

from modelscope.fileio import File
from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import InputModel, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['SpeakerChangeLocatingPipeline']


@PIPELINES.register_module(
    Tasks.speaker_diarization, module_name=Pipelines.speaker_change_locating)
class SpeakerChangeLocatingPipeline(Pipeline):
    """Speaker Change Locating Inference Pipeline
    use `model` to create a speaker change Locating pipeline.

    Args:
        model (SpeakerChangeLocatingPipeline): A model instance, or a model local dir, or a model id in the model hub.
        kwargs (dict, `optional`):
            Extra kwargs passed into the pipeline's constructor.
    Example:
    >>> from modelscope.pipelines import pipeline
    >>> from modelscope.utils.constant import Tasks
    >>> p = pipeline(
    >>>    task=Tasks.speaker_diarization, model='damo/speech_campplus-transformer_scl_zh-cn_16k-common')
    >>> print(p(audio))

    """

    def __init__(self, model: InputModel, **kwargs):
        """use `model` to create a speaker change Locating pipeline for prediction
        Args:
            model (str): a valid offical model id
        """
        super().__init__(model=model, **kwargs)
        self.model_config = self.model.model_config
        self.anchor_size = self.model_config['anchor_size']

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        embds: Union[list, np.ndarray] = None,
        output_res=False,
    ):
        if embds is not None:
            assert len(embds) == 2
            assert isinstance(embds[0], np.ndarray) and isinstance(
                embds[1], np.ndarray)
            assert embds[0].shape == (
                self.anchor_size, ) and embds[1].shape == (self.anchor_size, )
        else:
            embd1 = np.zeros(self.anchor_size // 2)
            embd2 = np.ones(self.anchor_size - self.anchor_size // 2)
            embd3 = np.ones(self.anchor_size // 2)
            embd4 = np.zeros(self.anchor_size - self.anchor_size // 2)
            embds = [
                np.stack([embd1, embd2], axis=1).flatten(),
                np.stack([embd3, embd4], axis=1).flatten(),
            ]
        if isinstance(embds, list):
            anchors = np.stack(embds, axis=0)
        anchors = torch.from_numpy(anchors).unsqueeze(0).float()

        output = self.preprocess(audio)
        output = self.forward(output, anchors)
        output, p = self.postprocess(output)

        if output_res:
            return output, p
        else:
            return output

    def forward(self, input: torch.Tensor, anchors: torch.Tensor):
        output = self.model(input, anchors)
        return output

    def postprocess(self, input: torch.Tensor):
        predict = np.where(np.diff(input.argmax(-1).numpy()))
        try:
            predict = predict[0][0] * 0.01 + 0.02
            predict = round(predict, 2)
            return {
                OutputKeys.TEXT: f'The change point is at {predict}s.'
            }, predict
        except Exception:
            return {OutputKeys.TEXT: 'No change point is found.'}, None

    def preprocess(self, input: Union[str, np.ndarray]) -> torch.Tensor:
        if isinstance(input, str):
            file_bytes = File.read(input)
            data, fs = sf.read(io.BytesIO(file_bytes), dtype='float32')
            if len(data.shape) == 2:
                data = data[:, 0]
            data = torch.from_numpy(data).unsqueeze(0)
            if fs != self.model_config['sample_rate']:
                logger.warning(
                    'The sample rate of audio is not %d, resample it.'
                    % self.model_config['sample_rate'])
                data, fs = torchaudio.sox_effects.apply_effects_tensor(
                    data,
                    fs,
                    effects=[['rate',
                              str(self.model_config['sample_rate'])]])
        elif isinstance(input, np.ndarray):
            if input.dtype in ['int16', 'int32', 'int64']:
                input = (input / (1 << 15)).astype('float32')
            else:
                input = input.astype('float32')
            data = torch.from_numpy(input)
            if len(data.shape) == 1:
                data = data.unsqueeze(0)
        else:
            raise ValueError(
                'modelscope error: The input type is restricted to audio file address and numpy array.'
            )
        return data
