# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, List

import numpy as np

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.audio.tts import SambertHifigan
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, InputModel, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Fields, Tasks

__all__ = ['TextToSpeechSambertHifiganPipeline']


@PIPELINES.register_module(
    Tasks.text_to_speech, module_name=Pipelines.sambert_hifigan_tts)
class TextToSpeechSambertHifiganPipeline(Pipeline):

    def __init__(self, model: InputModel, **kwargs):
        """use `model` to create a text-to-speech pipeline for prediction

        Args:
            model (SambertHifigan or str): a model instance or valid offical model id
        """
        super().__init__(model=model, **kwargs)

    def forward(self, input: str, **forward_params) -> Dict[str, bytes]:
        """synthesis text from inputs with pipeline
        Args:
            input (str): text to synthesis
            forward_params: valid param is 'voice' used to setting speaker vocie
        Returns:
            Dict[str, np.ndarray]: {OutputKeys.OUTPUT_PCM : np.ndarray(16bit pcm data)}
        """
        output_wav = self.model.forward(input, forward_params.get('voice'))
        return {OutputKeys.OUTPUT_WAV: output_wav}

    def postprocess(self, inputs: Dict[str, Any],
                    **postprocess_params) -> Dict[str, Any]:
        return inputs

    def preprocess(self, inputs: Input, **preprocess_params) -> Dict[str, Any]:
        return inputs

    def _sanitize_parameters(self, **pipeline_parameters):
        return {}, pipeline_parameters, {}
