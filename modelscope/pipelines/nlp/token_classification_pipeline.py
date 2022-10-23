# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Optional, Union

import torch

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import (Preprocessor,
                                      TokenClassificationPreprocessor)
from modelscope.utils.constant import Tasks

__all__ = ['TokenClassificationPipeline']


@PIPELINES.register_module(
    Tasks.token_classification, module_name=Pipelines.part_of_speech)
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
        assert isinstance(model, str) or isinstance(model, Model), \
            'model must be a single str or Model'
        model = model if isinstance(model,
                                    Model) else Model.from_pretrained(model)
        if preprocessor is None:
            preprocessor = TokenClassificationPreprocessor(
                model.model_dir,
                sequence_length=kwargs.pop('sequence_length', 128))
        model.eval()
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        if hasattr(model, 'id2label'):
            self.id2label = getattr(model, 'id2label')
        else:
            model_config = getattr(model, 'config')
            self.id2label = getattr(model_config, 'id2label')

        assert self.id2label is not None, 'Cannot convert id to the original label, please pass in the mapping ' \
                                          'as a parameter or make sure the preprocessor has the attribute.'

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
            inputs (Dict[str, Any]): _description_

        Returns:
            Dict[str, str]: the prediction results
        """

        pred_list = inputs['predictions']
        labels = []
        for pre in pred_list:
            labels.append(self.id2label[pre])
        labels = labels[1:-1]
        chunks = []
        tags = []
        chunk = ''
        assert len(inputs['text']) == len(labels)
        for token, label in zip(inputs['text'], labels):
            if label[0] == 'B' or label[0] == 'I':
                chunk += token
            else:
                chunk += token
                chunks.append(chunk)
                chunk = ''
                tags.append(label.split('-')[-1])
        if chunk:
            chunks.append(chunk)
            tags.append(label.split('-')[-1])
        pos_result = []
        seg_result = ' '.join(chunks)
        for chunk, tag in zip(chunks, tags):
            pos_result.append({OutputKeys.WORD: chunk, OutputKeys.LABEL: tag})
        outputs = {
            OutputKeys.OUTPUT: seg_result,
            OutputKeys.LABELS: pos_result
        }
        return outputs
