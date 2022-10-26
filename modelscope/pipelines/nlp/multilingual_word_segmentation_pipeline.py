# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Optional, Union

import torch

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import (Preprocessor,
                                      TokenClassificationPreprocessor,
                                      WordSegmentationPreprocessorThai)
from modelscope.utils.constant import Tasks

__all__ = [
    'MultilingualWordSegmentationPipeline', 'WordSegmentationThaiPipeline'
]


@PIPELINES.register_module(
    Tasks.word_segmentation,
    module_name=Pipelines.multilingual_word_segmentation)
class MultilingualWordSegmentationPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 **kwargs):
        """Use `model` and `preprocessor` to create a nlp word segmentation pipeline for prediction

        Args:
            model (str or Model): Supply either a local model dir which supported word segmentation task, or a
            model id from the model hub, or a torch model instance.
            preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for
            the model if supplied.
            sequence_length: Max sequence length in the user's custom scenario. 512 will be used as a default value.

            To view other examples plese check the tests/pipelines/test_multilingual_word_segmentation.py.
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
                **super().forward(inputs, **forward_params), OutputKeys.TEXT:
                text
            }

    def postprocess(self, inputs: Dict[str, Any],
                    **postprocess_params) -> Dict[str, str]:
        text = inputs['text']
        offset_mapping = [x.cpu().tolist() for x in inputs['offset_mapping']]
        labels = [
            self.id2label[x]
            for x in inputs['predictions'].squeeze(0).cpu().numpy()
        ]
        entities = []
        entity = {}
        for label, offsets in zip(labels, offset_mapping):
            if label[0] in 'BS':
                if entity:
                    entity['span'] = text[entity['start']:entity['end']]
                    entities.append(entity)
                entity = {
                    'type': label[2:],
                    'start': offsets[0],
                    'end': offsets[1]
                }
            if label[0] in 'IES':
                if entity:
                    entity['end'] = offsets[1]
            if label[0] in 'ES':
                if entity:
                    entity['span'] = text[entity['start']:entity['end']]
                    entities.append(entity)
                    entity = {}
        if entity:
            entity['span'] = text[entity['start']:entity['end']]
            entities.append(entity)

        word_segments = [entity['span'] for entity in entities]
        outputs = {OutputKeys.OUTPUT: word_segments, OutputKeys.LABELS: []}

        return outputs


@PIPELINES.register_module(
    Tasks.word_segmentation, module_name=Pipelines.word_segmentation_thai)
class WordSegmentationThaiPipeline(MultilingualWordSegmentationPipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 **kwargs):
        model = model if isinstance(model,
                                    Model) else Model.from_pretrained(model)
        if preprocessor is None:
            preprocessor = WordSegmentationPreprocessorThai(
                model.model_dir,
                sequence_length=kwargs.pop('sequence_length', 512))
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)

    def postprocess(self, inputs: Dict[str, Any],
                    **postprocess_params) -> Dict[str, str]:
        outputs = super().postprocess(inputs, **postprocess_params)
        word_segments = outputs[OutputKeys.OUTPUT]
        word_segments = [seg.replace(' ', '') for seg in word_segments]

        return {OutputKeys.OUTPUT: word_segments, OutputKeys.LABELS: []}
