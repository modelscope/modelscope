# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Any, Dict

from modelscope.metainfo import Models
from modelscope.models.base import Model
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks

import wenetruntime as wenet

__all__ = ['WeNetAutomaticSpeechRecognition']


@MODELS.register_module(
    Tasks.auto_speech_recognition, module_name=Models.wenet_asr)
class WeNetAutomaticSpeechRecognition(Model):

    def __init__(self, model_dir: str, am_model_name: str,
                 model_config: Dict[str, Any], *args, **kwargs):
        """initialize the info of model.

        Args:
            model_dir (str): the model path.
            am_model_name (str): the am model name from configuration.json
            model_config (Dict[str, Any]): the detail config about model from configuration.json
        """
        super().__init__(model_dir, am_model_name, model_config, *args,
                         **kwargs)
        self.model_cfg = {
            # the recognition model dir path
            'model_dir': model_dir,
            # the recognition model config dict
            'model_config': model_config
        }
        self.decoder = None

    def forward(self) -> Dict[str, Any]:
        """preload model and return the info of the model
        """
        model_dir = self.model_cfg['model_dir']
        self.decoder = wenet.Decoder(model_dir, lang='chs')

        return self.model_cfg
