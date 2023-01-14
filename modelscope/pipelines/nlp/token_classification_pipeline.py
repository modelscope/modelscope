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
from modelscope.utils.constant import Tasks
from modelscope.utils.tensor_utils import (torch_nested_detach,
                                           torch_nested_numpify)

__all__ = ['TokenClassificationPipeline']


@PIPELINES.register_module(
    Tasks.token_classification, module_name=Pipelines.token_classification)
@PIPELINES.register_module(
    Tasks.token_classification, module_name=Pipelines.part_of_speech)
@PIPELINES.register_module(
    Tasks.token_classification, module_name=Pipelines.word_segmentation)
@PIPELINES.register_module(
    Tasks.token_classification, module_name=Pipelines.named_entity_recognition)
@PIPELINES.register_module(
    Tasks.part_of_speech, module_name=Pipelines.part_of_speech)
class TokenClassificationPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 config_file: str = None,
                 device: str = 'gpu',
                 auto_collate=True,
                 sequence_length=512,
                 **kwargs):
        """use `model` and `preprocessor` to create a token classification pipeline for prediction

        Args:
            model (str or Model): A model instance or a model local dir or a model id in the model hub.
            preprocessor (Preprocessor): a preprocessor instance, must not be None.
            kwargs (dict, `optional`):
                Extra kwargs passed into the preprocessor's constructor.
        """
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate)

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
            return {
                **self.model(**inputs, **forward_params), OutputKeys.TEXT: text
            }

    def postprocess(self, inputs: Dict[str, Any],
                    **postprocess_params) -> Dict[str, Any]:
        """Process the prediction results

        Args:
            inputs (Dict[str, Any]): should be tensors from model

        Returns:
            Dict[str, Any]: the prediction results
        """
        chunks = self._chunk_process(inputs, **postprocess_params)
        return {OutputKeys.OUTPUT: chunks}

    def _chunk_process(self, inputs: Dict[str, Any],
                       **postprocess_params) -> List:
        """process the prediction results and output as chunks

        Args:
            inputs (Dict[str, Any]): should be tensors from model

        Returns:
            List: The output chunks
        """
        text = inputs['text']
        # TODO post_process does not support batch for now.
        if OutputKeys.PREDICTIONS not in inputs:
            logits = inputs[OutputKeys.LOGITS]
            if len(logits.shape) == 3:
                logits = logits[0]
            predictions = torch.argmax(logits, dim=-1)
        else:
            predictions = inputs[OutputKeys.PREDICTIONS]
            if len(predictions.shape) == 2:
                predictions = predictions[0]

        offset_mapping = inputs['offset_mapping']
        if len(offset_mapping.shape) == 3:
            offset_mapping = offset_mapping[0]

        label_mask = inputs.get('label_mask')
        if label_mask is not None:
            masked_lengths = label_mask.sum(-1).long().cpu().item()
            offset_mapping = torch.narrow(
                offset_mapping, 0, 0,
                masked_lengths)  # index_select only move loc, not resize

            if len(label_mask.shape) == 2:
                label_mask = label_mask[0]
            predictions = predictions.masked_select(label_mask)

        offset_mapping = torch_nested_numpify(
            torch_nested_detach(offset_mapping))
        predictions = torch_nested_numpify(torch_nested_detach(predictions))
        labels = [self.id2label[x] for x in predictions]

        chunks = []
        chunk = {}
        for label, offsets in zip(labels, offset_mapping):
            if label[0] in 'BS':
                if chunk:
                    chunk['span'] = text[chunk['start']:chunk['end']]
                    chunks.append(chunk)
                chunk = {
                    'type': label[2:],
                    'start': offsets[0],
                    'end': offsets[1]
                }
            if label[0] in 'I':
                if not chunk:
                    chunk = {
                        'type': label[2:],
                        'start': offsets[0],
                        'end': offsets[1]
                    }
            if label[0] in 'E':
                if not chunk:
                    chunk = {
                        'type': label[2:],
                        'start': offsets[0],
                        'end': offsets[1]
                    }
            if label[0] in 'IES':
                if chunk:
                    chunk['end'] = offsets[1]

            if label[0] in 'ES':
                if chunk:
                    chunk['span'] = text[chunk['start']:chunk['end']]
                    chunks.append(chunk)
                    chunk = {}

        if chunk:
            chunk['span'] = text[chunk['start']:chunk['end']]
            chunks.append(chunk)

        return chunks
