# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Any, Dict, Union

import numpy as np
import sentencepiece
from fasttext import load_model
from fasttext.FastText import _FastText

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import ModelFile, Tasks

__all__ = ['FasttextSequenceClassificationPipeline']


def sentencepiece_tokenize(sp_model, sent):
    tokens = []
    for t in sp_model.EncodeAsPieces(sent):
        s = t.strip()
        if s:
            tokens.append(s)
    return ' '.join(tokens)


@PIPELINES.register_module(
    Tasks.text_classification, module_name=Pipelines.domain_classification)
class FasttextSequenceClassificationPipeline(Pipeline):

    def __init__(self, model: Union[str, _FastText], **kwargs):
        """use `model` and `preprocessor` to create a nlp text classification pipeline for prediction

        Args:
            model: A model directory including model.bin and spm.model
        """
        super().__init__(model=model)
        model_file = os.path.join(model, ModelFile.TORCH_MODEL_BIN_FILE)
        spm_file = os.path.join(model, 'sentencepiece.model')
        assert os.path.isdir(model) and os.path.exists(model_file) and os.path.exists(spm_file), \
            '`model` should be a directory contains `model.bin` and `sentencepiece.model`'
        self.model = load_model(model_file)
        self.spm = sentencepiece.SentencePieceProcessor()
        self.spm.Load(spm_file)

    def preprocess(self, inputs: str) -> Dict[str, Any]:
        text = inputs.strip()
        text_sp = sentencepiece_tokenize(self.spm, text)
        return {'text_sp': text_sp, 'text': text}

    def forward(self,
                inputs: Dict[str, Any],
                topk: int = None) -> Dict[str, Any]:
        if topk is None:
            topk = inputs.get('topk', -1)
        label, probs = self.model.predict(inputs['text_sp'], k=topk)
        label = [x.replace('__label__', '') for x in label]
        result = {
            OutputKeys.LABEL: label[0],
            OutputKeys.SCORE: probs[0],
            OutputKeys.LABELS: label,
            OutputKeys.SCORES: probs
        }
        return result

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
