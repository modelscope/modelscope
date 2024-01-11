# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, List, Sequence, Tuple, Union

import json
import yaml

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.audio.audio_utils import (generate_scp_from_url,
                                                update_local_model)
from modelscope.utils.constant import Frameworks, ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['FunASRPipeline']


@PIPELINES.register_module(
    Tasks.auto_speech_recognition, module_name=Pipelines.funasr_pipeline)
@PIPELINES.register_module(
    Tasks.voice_activity_detection, module_name=Pipelines.funasr_pipeline)
@PIPELINES.register_module(
    Tasks.language_score_prediction, module_name=Pipelines.funasr_pipeline)
@PIPELINES.register_module(
    Tasks.punctuation, module_name=Pipelines.funasr_pipeline)
@PIPELINES.register_module(
    Tasks.speaker_diarization, module_name=Pipelines.funasr_pipeline)
@PIPELINES.register_module(
    Tasks.speaker_verification, module_name=Pipelines.funasr_pipeline)
@PIPELINES.register_module(
    Tasks.speech_separation, module_name=Pipelines.funasr_pipeline)
@PIPELINES.register_module(
    Tasks.speech_timestamp, module_name=Pipelines.funasr_pipeline)
@PIPELINES.register_module(
    Tasks.emotion_recognition, module_name=Pipelines.funasr_pipeline)
class FunASRPipeline(Pipeline):
    """Voice Activity Detection Inference Pipeline
    use `model` to create a Voice Activity Detection pipeline.

    Args:
        model: A model instance, or a model local dir, or a model id in the model hub.
        kwargs (dict, `optional`):
            Extra kwargs passed into the preprocessor's constructor.

    Example:
        >>> from modelscope.pipelines import pipeline
        >>> p = pipeline(
        >>>    task=Tasks.voice_activity_detection, model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch')
        >>> audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example.pcm'
        >>> print(p(audio_in))

    """

    def __init__(self, model: Union[Model, str] = None, **kwargs):
        """use `model` to create an vad pipeline for prediction
        """
        super().__init__(model=model, **kwargs)

    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Decoding the input audios
        Args:
            input('str' or 'bytes'):
        Return:
            a list of dictionary of result.
        """

        output = self.model(*args, **kwargs)

        return output
