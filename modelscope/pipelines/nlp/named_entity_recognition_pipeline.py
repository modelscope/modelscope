from typing import Any, Dict, Optional, Union

import torch

from ...metainfo import Pipelines
from ...models import Model
from ...models.nlp import TransformerCRFForNamedEntityRecognition
from ...outputs import OutputKeys
from ...preprocessors import NERPreprocessor
from ...utils.constant import Tasks
from ..base import Pipeline, Tensor
from ..builder import PIPELINES

__all__ = ['NamedEntityRecognitionPipeline']


@PIPELINES.register_module(
    Tasks.named_entity_recognition,
    module_name=Pipelines.named_entity_recognition)
class NamedEntityRecognitionPipeline(Pipeline):

    def __init__(self,
                 model: Union[TransformerCRFForNamedEntityRecognition, str],
                 preprocessor: Optional[NERPreprocessor] = None,
                 **kwargs):

        model = model if isinstance(model,
                                    TransformerCRFForNamedEntityRecognition
                                    ) else Model.from_pretrained(model)
        if preprocessor is None:
            preprocessor = NERPreprocessor(model.model_dir)
        model.eval()
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.tokenizer = preprocessor.tokenizer
        self.config = model.config
        assert len(self.config.id2label) > 0
        self.id2label = self.config.id2label

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            return super().forward(inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Any],
                    **postprocess_params) -> Dict[str, str]:
        text = inputs['text']
        offset_mapping = inputs['offset_mapping']
        labels = [self.id2label[x] for x in inputs['predicts']]
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
        outputs = {OutputKeys.OUTPUT: entities}

        return outputs
