# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Optional, Union

import torch

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import (Preprocessor,
                                      TokenClassificationPreprocessor)
from modelscope.utils.constant import Tasks
from modelscope.utils.tensor_utils import (torch_nested_detach,
                                           torch_nested_numpify)

__all__ = ['NamedEntityRecognitionPipeline']


@PIPELINES.register_module(
    Tasks.named_entity_recognition,
    module_name=Pipelines.named_entity_recognition)
class NamedEntityRecognitionPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 **kwargs):
        """Use `model` and `preprocessor` to create a nlp NER pipeline for prediction

        Args:
            model (str or Model): Supply either a local model dir which supported NER task, or a
            model id from the model hub, or a torch model instance.
            preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for
            the model if supplied.
            sequence_length: Max sequence length in the user's custom scenario. 512 will be used as a default value.

            Example:
            >>> from modelscope.pipelines import pipeline
            >>> pipeline_ins = pipeline(task='named-entity-recognition',
            >>>        model='damo/nlp_raner_named-entity-recognition_chinese-base-news')
            >>> input = '这与温岭市新河镇的一个神秘的传说有关。'
            >>> print(pipeline_ins(input))

            To view other examples plese check the tests/pipelines/test_named_entity_recognition.py.
        """

        model = model if isinstance(model,
                                    Model) else Model.from_pretrained(model)
        if preprocessor is None:
            preprocessor = TokenClassificationPreprocessor(
                model.model_dir,
                sequence_length=kwargs.pop('sequence_length', 512))
        model.eval()
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.tokenizer = preprocessor.tokenizer
        self.config = model.config
        assert len(self.config.id2label) > 0
        self.id2label = self.config.id2label

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
        if OutputKeys.PREDICTIONS not in inputs:
            logits = inputs[OutputKeys.LOGITS]
            predictions = torch.argmax(logits[0], dim=-1)
        else:
            predictions = inputs[OutputKeys.PREDICTIONS].squeeze(
                0).cpu().numpy()
        predictions = torch_nested_numpify(torch_nested_detach(predictions))
        offset_mapping = [x.cpu().tolist() for x in inputs['offset_mapping']]

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

        # for cws output
        if len(chunks) > 0 and chunks[0]['type'] == 'cws':
            spans = [
                chunk['span'] for chunk in chunks if chunk['span'].strip()
            ]
            seg_result = ' '.join(spans)
            outputs = {OutputKeys.OUTPUT: seg_result, OutputKeys.LABELS: []}

        # for ner outpus
        else:
            outputs = {OutputKeys.OUTPUT: chunks}
        return outputs
