# Copyright (c) Alibaba, Inc. and its affiliates.

import io
import os
from typing import Union

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

__all__ = ['LanguageRecognitionPipeline']


@PIPELINES.register_module(
    Tasks.speech_language_recognition,
    module_name=Pipelines.speech_language_recognition_eres2net)
class LanguageRecognitionPipeline(Pipeline):
    """Language Recognition Inference Pipeline
    use `model` to create a Language Recognition pipeline.

    Args:
        model (LanguageRecognitionPipeline): A model instance, or a model local dir, or a model id in the model hub.
        kwargs (dict, `optional`):
            Extra kwargs passed into the pipeline's constructor.
    Example:
    >>> from modelscope.pipelines import pipeline
    >>> from modelscope.utils.constant import Tasks
    >>> p = pipeline(
    >>>    task=Tasks.speech_language_recognition, model='damo/speech_eres2net_base_lre_en-cn_16k')
    >>> print(p(audio_in))

    """

    def __init__(self, model: InputModel, **kwargs):
        """use `model` to create a Language Recognition pipeline for prediction
        Args:
            model (str): a valid offical model id
        """
        super().__init__(model=model, **kwargs)
        self.model_config = self.model.model_config
        self.languages = self.model_config['languages']

    def __call__(self,
                 in_audios: Union[str, list, np.ndarray],
                 out_file: str = None):
        wavs = self.preprocess(in_audios)
        scores, results = self.forward(wavs)
        outputs = self.postprocess(results, scores, in_audios, out_file)
        return outputs

    def forward(self, inputs: list):
        scores = []
        results = []
        for x in inputs:
            score, result = self.model(x)
            scores.append(score.tolist())
            results.append(result.item())
        return scores, results

    def postprocess(self,
                    inputs: list,
                    scores: list,
                    in_audios: Union[str, list, np.ndarray],
                    out_file=None):
        if isinstance(in_audios, str):
            output = {
                OutputKeys.TEXT: self.languages[inputs[0]],
                OutputKeys.SCORE: scores
            }
        else:
            output = {
                OutputKeys.TEXT: [self.languages[i] for i in inputs],
                OutputKeys.SCORE: scores
            }
            if out_file is not None:
                out_lines = []
                for i, audio in enumerate(in_audios):
                    if isinstance(audio, str):
                        audio_id = os.path.basename(audio).rsplit('.', 1)[0]
                    else:
                        audio_id = i
                    out_lines.append('%s %s\n' %
                                     (audio_id, self.languages[inputs[i]]))
                with open(out_file, 'w') as f:
                    for i in out_lines:
                        f.write(i)
        return output

    def preprocess(self, inputs: Union[str, list, np.ndarray]):
        output = []
        if isinstance(inputs, str):
            file_bytes = File.read(inputs)
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
            data = data.squeeze(0)
            output.append(data)
        else:
            for i in range(len(inputs)):
                if isinstance(inputs[i], str):
                    file_bytes = File.read(inputs[i])
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
                            effects=[[
                                'rate',
                                str(self.model_config['sample_rate'])
                            ]])
                    data = data.squeeze(0)
                elif isinstance(inputs[i], np.ndarray):
                    assert len(
                        inputs[i].shape
                    ) == 1, 'modelscope error: Input array should be [N, T]'
                    data = inputs[i]
                    if data.dtype in ['int16', 'int32', 'int64']:
                        data = (data / (1 << 15)).astype('float32')
                    else:
                        data = data.astype('float32')
                    data = torch.from_numpy(data)
                else:
                    raise ValueError(
                        'modelscope error: The input type is restricted to audio address and nump array.'
                    )
                output.append(data)
        return output
