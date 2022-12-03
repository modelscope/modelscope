# Copyright (c) Alibaba, Inc. and its affiliates.

import io
import os
from typing import Any, Dict

import torch
from transformers import XLMRobertaTokenizer

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import ModelFile, Tasks

__all__ = ['TranslationQualityEstimationPipeline']


@PIPELINES.register_module(
    Tasks.sentence_similarity,
    module_name=Pipelines.translation_quality_estimation)
class TranslationQualityEstimationPipeline(Pipeline):

    def __init__(self, model: str, device: str = 'gpu', **kwargs):
        super().__init__(model=model, device=device)
        model_file = os.path.join(self.model, ModelFile.TORCH_MODEL_FILE)
        with open(model_file, 'rb') as f:
            buffer = io.BytesIO(f.read())
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.model)
        self.model = torch.jit.load(
            buffer, map_location=self.device).to(self.device)

    def preprocess(self, inputs: Dict[str, Any]):
        src_text = inputs['source_text'].strip()
        tgt_text = inputs['target_text'].strip()
        encoded_inputs = self.tokenizer.batch_encode_plus(
            [[src_text, tgt_text]],
            return_tensors='pt',
            padding=True,
            truncation=True)
        input_ids = encoded_inputs['input_ids'].to(self.device)
        attention_mask = encoded_inputs['attention_mask'].to(self.device)
        inputs.update({
            'input_ids': input_ids,
            'attention_mask': attention_mask
        })
        return inputs

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if 'input_ids' not in inputs:
            inputs = self.preprocess(inputs)
        res = self.model(inputs['input_ids'], inputs['attention_mask'])
        result = {
            OutputKeys.LABELS: '-1',
            OutputKeys.SCORES: res[0].detach().squeeze().tolist()
        }
        return result

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): input data dict

        Returns:
            Dict[str, str]: the prediction results
        """
        return inputs
