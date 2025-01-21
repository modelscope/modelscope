# Copyright (c) Alibaba, Inc. and its affiliates.

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import ModelFile, Tasks
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
        self.sequence_length = sequence_length

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

        return_prob = postprocess_params.pop('return_prob', True)
        if return_prob:
            if OutputKeys.LOGITS in inputs:
                logits = inputs[OutputKeys.LOGITS]
                if len(logits.shape) == 3:
                    logits = logits[0]
                probs = torch_nested_numpify(
                    torch_nested_detach(logits.softmax(-1)))
            else:
                return_prob = False

        chunks = []
        chunk = {}
        for i, (label, offsets) in enumerate(zip(labels, offset_mapping)):
            if label[0] in 'BS':
                if chunk:
                    chunk['span'] = text[chunk['start']:chunk['end']]
                    chunks.append(chunk)
                chunk = {
                    'type': label[2:],
                    'start': offsets[0],
                    'end': offsets[1]
                }
                if return_prob:
                    chunk['prob'] = probs[i][predictions[i]]
            if label[0] in 'I':
                if not chunk:
                    chunk = {
                        'type': label[2:],
                        'start': offsets[0],
                        'end': offsets[1]
                    }
                    if return_prob:
                        chunk['prob'] = probs[i][predictions[i]]
            if label[0] in 'E':
                if not chunk:
                    chunk = {
                        'type': label[2:],
                        'start': offsets[0],
                        'end': offsets[1]
                    }
                    if return_prob:
                        chunk['prob'] = probs[i][predictions[i]]
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

    def _process_single(self, input: Input, *args, **kwargs) -> Dict[str, Any]:
        split_max_length = kwargs.pop('split_max_length',
                                      0)  # default: no split
        if split_max_length <= 0:
            return super()._process_single(input, *args, **kwargs)
        else:
            split_texts, index_mapping = self._auto_split([input],
                                                          split_max_length)
            outputs = []
            for text in split_texts:
                outputs.append(super()._process_single(text, *args, **kwargs))
            return self._auto_join(outputs, index_mapping)[0]

    def _process_batch(self, input: List[Input], batch_size: int, *args,
                       **kwargs) -> List[Dict[str, Any]]:
        split_max_length = kwargs.pop('split_max_length',
                                      0)  # default: no split
        if split_max_length <= 0:
            return super()._process_batch(
                input, batch_size=batch_size, *args, **kwargs)
        else:
            split_texts, index_mapping = self._auto_split(
                input, split_max_length)
            outputs = super()._process_batch(
                split_texts, batch_size=batch_size, *args, **kwargs)
            return self._auto_join(outputs, index_mapping)

    def _auto_split(self, input_texts: List[str], split_max_length: int):
        split_texts = []
        index_mapping = {}
        new_idx = 0
        for raw_idx, text in enumerate(input_texts):
            if len(text) < split_max_length:
                split_texts.append(text)
                index_mapping[new_idx] = (raw_idx, 0)
                new_idx += 1
            else:
                n_split = math.ceil(len(text) / split_max_length)
                for i in range(n_split):
                    offset = i * split_max_length
                    split_texts.append(text[offset:offset + split_max_length])
                    index_mapping[new_idx] = (raw_idx, offset)
                    new_idx += 1
        return split_texts, index_mapping

    def _auto_join(
            self, outputs: List[Dict[str, Any]],
            index_mapping: Dict[int, Tuple[int, int]]) -> List[Dict[str, Any]]:
        joined_outputs = []
        for idx, output in enumerate(outputs):
            raw_idx, offset = index_mapping[idx]
            if raw_idx >= len(joined_outputs):
                joined_outputs.append(output)
            else:
                for chunk in output[OutputKeys.OUTPUT]:
                    chunk['start'] += offset
                    chunk['end'] += offset
                    joined_outputs[raw_idx][OutputKeys.OUTPUT].append(chunk)
        return joined_outputs
