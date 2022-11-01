# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Optional, Union

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
                 **kwargs):
        """use `model` and `preprocessor` to create a token classification pipeline for prediction

        Args:
            model (str or Model): A model instance or a model local dir or a model id in the model hub.
            preprocessor (Preprocessor): a preprocessor instance, must not be None.
        """
        model = Model.from_pretrained(model) if isinstance(model,
                                                           str) else model

        if preprocessor is None:
            preprocessor = Preprocessor.from_pretrained(
                model.model_dir,
                sequence_length=kwargs.pop('sequence_length', 128))
        model.eval()
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.id2label = kwargs.get('id2label')
        if self.id2label is None and hasattr(self.preprocessor, 'id2label'):
            self.id2label = self.preprocessor.id2label

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        text = inputs.pop(OutputKeys.TEXT)
        with torch.no_grad():
            return {
                **self.model(**inputs, **forward_params), OutputKeys.TEXT: text
            }

    def postprocess(self, inputs: Dict[str, Any],
                    **postprocess_params) -> Dict[str, str]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): should be tensors from model

        Returns:
            Dict[str, str]: the prediction results
        """
        text = inputs['text']
        if not hasattr(inputs, 'predictions'):
            logits = inputs[OutputKeys.LOGITS]
            predictions = torch.argmax(logits[0], dim=-1)
        else:
            predictions = inputs[OutputKeys.PREDICTIONS].squeeze(
                0).cpu().numpy()
        predictions = torch_nested_numpify(torch_nested_detach(predictions))
        offset_mapping = [x.cpu().tolist() for x in inputs['offset_mapping']]

        labels = [self.id2label[x] for x in predictions]
        if len(labels) > len(offset_mapping):
            labels = labels[1:-1]
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

        # for cws outputs
        if len(chunks) > 0 and chunks[0]['type'] == 'cws':
            spans = [
                chunk['span'] for chunk in chunks if chunk['span'].strip()
            ]
            seg_result = ' '.join(spans)
            outputs = {OutputKeys.OUTPUT: seg_result}

        # for ner outputs
        else:
            outputs = {OutputKeys.OUTPUT: chunks}
        return outputs
