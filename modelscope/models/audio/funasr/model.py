# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Any, Dict

import json
from funasr import AutoModel

from modelscope.metainfo import Models
from modelscope.models.base import Model
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Frameworks, Tasks

__all__ = ['GenericFunASR']


@MODELS.register_module(
    Tasks.auto_speech_recognition, module_name=Models.funasr)
@MODELS.register_module(
    Tasks.voice_activity_detection, module_name=Models.funasr)
@MODELS.register_module(
    Tasks.language_score_prediction, module_name=Models.funasr)
@MODELS.register_module(Tasks.punctuation, module_name=Models.funasr)
@MODELS.register_module(Tasks.speaker_diarization, module_name=Models.funasr)
@MODELS.register_module(Tasks.speaker_verification, module_name=Models.funasr)
@MODELS.register_module(Tasks.speech_separation, module_name=Models.funasr)
@MODELS.register_module(Tasks.speech_timestamp, module_name=Models.funasr)
@MODELS.register_module(Tasks.emotion_recognition, module_name=Models.funasr)
class GenericFunASR(Model):

    def __init__(self, model_dir, *args, **kwargs):
        """initialize the info of model.

        Args:
            model_dir (str): the model path.
            am_model_name (str): the am model name from configuration.json
            model_config (Dict[str, Any]): the detail config about model from configuration.json
        """
        super().__init__(model_dir, *args, **kwargs)
        model_cfg = json.loads(
            open(os.path.join(model_dir, 'configuration.json')).read())
        if 'vad_model' not in kwargs and 'vad_model' in model_cfg:
            kwargs['vad_model'] = model_cfg['vad_model']
            kwargs['vad_model_revision'] = model_cfg.get(
                'vad_model_revision', None)
        if 'punc_model' not in kwargs and 'punc_model' in model_cfg:
            kwargs['punc_model'] = model_cfg['punc_model']
            kwargs['punc_model_revision'] = model_cfg.get(
                'punc_model_revision', None)
        if 'spk_model' not in kwargs and 'spk_model' in model_cfg:
            kwargs['spk_model'] = model_cfg['spk_model']
            kwargs['spk_model_revision'] = model_cfg.get(
                'spk_model_revision', None)

        self.model = AutoModel(model=model_dir, **kwargs)

    def forward(self, *args, **kwargs):
        """preload model and return the info of the model
        """

        output = self.model.generate(*args, **kwargs)
        return output
