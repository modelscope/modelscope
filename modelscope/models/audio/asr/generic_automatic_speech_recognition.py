# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Any, Dict

from modelscope.metainfo import Models
from modelscope.models.base import Model
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Frameworks, Tasks

__all__ = ['GenericAutomaticSpeechRecognition']


@MODELS.register_module(
    Tasks.auto_speech_recognition, module_name=Models.generic_asr)
@MODELS.register_module(
    Tasks.voice_activity_detection, module_name=Models.generic_asr)
@MODELS.register_module(Tasks.language_model, module_name=Models.generic_asr)
class GenericAutomaticSpeechRecognition(Model):

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
            'model_workspace': model_dir,
            # the am model name
            'am_model': am_model_name,
            # the am model file path
            'am_model_path': os.path.join(model_dir, am_model_name),
            # the recognition model config dict
            'model_config': model_config
        }

    def forward(self) -> Dict[str, Any]:
        """preload model and return the info of the model
        """
        if self.model_cfg['model_config']['type'] == Frameworks.tf:
            from easyasr import asr_inference_paraformer_tf
            if hasattr(asr_inference_paraformer_tf, 'preload'):
                model_workspace = self.model_cfg['model_workspace']
                model_path = os.path.join(model_workspace,
                                          self.model_cfg['am_model'])
                vocab_path = os.path.join(
                    model_workspace,
                    self.model_cfg['model_config']['vocab_file'])
                sampled_ids = 'seq2seq/sampled_ids'
                sampled_lengths = 'seq2seq/sampled_lengths'
                if 'sampled_ids' in self.model_cfg['model_config']:
                    sampled_ids = self.model_cfg['model_config']['sampled_ids']
                if 'sampled_lengths' in self.model_cfg['model_config']:
                    sampled_lengths = self.model_cfg['model_config'][
                        'sampled_lengths']
                asr_inference_paraformer_tf.preload(
                    ngpu=1,
                    asr_model_file=model_path,
                    vocab_file=vocab_path,
                    sampled_ids=sampled_ids,
                    sampled_lengths=sampled_lengths)

        return self.model_cfg
