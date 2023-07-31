# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.tensor_utils import (torch_nested_detach,
                                           torch_nested_numpify)

logger = get_logger()


@PIPELINES.register_module(
    Tasks.speaker_diarization_semantic_speaker_turn_detection,
    module_name=Pipelines.speaker_diarization_semantic_speaker_turn_detection)
class SpeakerDiarizationSemanticSpeakerTurnDetectionPipeline(Pipeline):
    r"""The inference pipeline for Speaker Diarization Semantic Speaker-Turn Detection Task.

        Examples:
            >>> from modelscope.pipelines import pipeline
            >>> pipeline_ins = pipeline("speaker_diarization_semantic_speaker_turn_detection",
                            model="damo/speech_bert_semantic-spk-turn-detection-punc_speaker-diarization_chinese")
            >>> input_text = ""
            >>> print(pipeline_ins(input_text))
    """
    PUNC_LIST = ['。', '，', '？', '！']

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 config_file: str = None,
                 device: str = 'gpu',
                 auto_collate=True,
                 sequence_length=128,
                 **kwargs):
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate,
            compile=kwargs.pop('compile', False),
            compile_options=kwargs.pop('compile_options', {}))

        assert isinstance(self.model, Model), \
            f'please check whether model config exists in {ModelFile.CONFIGURATION}'

        if preprocessor is None:
            self.preprocessor = Preprocessor.from_pretrained(
                self.model.model_dir,
                sequence_length=sequence_length,
                **kwargs)
        self.model.eval()

        assert hasattr(self.preprocessor, 'id2label')
        self.id2label = self.preprocessor.id2label

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        text = inputs.pop(OutputKeys.TEXT)
        with torch.no_grad():
            outputs = self.model(**inputs, **forward_params)
            return {**outputs, OutputKeys.TEXT: text}

    def postprocess(self, inputs: Dict[str, Any],
                    **postprocess_params) -> Dict[str, Any]:
        r"""Precess the prediction results
            Args:
                inputs (dict[str, Any]): should be tensors from model

            Returns:
                Dict[str, Any]: the prediction results
        """
        text = inputs['text']

        if OutputKeys.PREDICTIONS not in inputs:
            logits = inputs[OutputKeys.LOGITS]
            if len(logits.shape) == 3:
                logits = logits[0]
            predictions = torch.argmax(logits, dim=-1)
        else:
            predictions = inputs[OutputKeys.PREDICTIONS]
            if len(predictions.shape) == 2:
                predictions = predictions[0]

        binary_prediction = []
        for i, ch in enumerate(text):
            if ch in self.PUNC_LIST:
                binary_prediction.append(0)
            else:
                binary_prediction.append(-100)

        result_text = ''
        for i, p in enumerate(predictions):
            if i >= len(text):
                continue
            result_text += text[i]
            if binary_prediction[i] != -100:
                binary_prediction[i] = p
                if p == 1:
                    result_text += '|'

        outputs = {
            'text': inputs['text'],
            'logits': inputs['logits'],
            'prediction': binary_prediction
        }
        return outputs
