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

    def forward(self, inputs: Dict[str, str]) -> Dict[str, np.ndarray]:
        """synthesis text from inputs with pipeline
        Args:
            inputs (Dict[str, str]): a dictionary that key is the name of
            certain testcase and value is the text to synthesis.
        Returns:
            Dict[str, np.ndarray]: a dictionary with key and value. The key
            is the same as inputs' key which is the label of the testcase
            and the value is the pcm audio data.
        """
        output_wav = {}
        for label, text in inputs.items():
            output_wav[label] = self.model.forward(text, inputs.get('voice'))
        return {OutputKeys.OUTPUT_PCM: output_wav}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def preprocess(self, inputs: Input, **preprocess_params) -> Dict[str, Any]:
        return inputs
