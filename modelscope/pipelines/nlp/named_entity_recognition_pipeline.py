# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Optional, Union

import torch

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.nlp import TokenClassificationPipeline
from modelscope.preprocessors import (NERPreprocessorThai, NERPreprocessorViet,
                                      Preprocessor,
                                      TokenClassificationPreprocessor)
from modelscope.utils.constant import Tasks
from modelscope.utils.tensor_utils import (torch_nested_detach,
                                           torch_nested_numpify)

__all__ = [
    'NamedEntityRecognitionPipeline', 'NamedEntityRecognitionThaiPipeline',
    'NamedEntityRecognitionVietPipeline'
]


@PIPELINES.register_module(
    Tasks.named_entity_recognition,
    module_name=Pipelines.named_entity_recognition)
class NamedEntityRecognitionPipeline(TokenClassificationPipeline):

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
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        if preprocessor is None:
            self.preprocessor = TokenClassificationPreprocessor(
                self.model.model_dir,
                sequence_length=kwargs.pop('sequence_length', 128))
        self.model.eval()
        self.id2label = kwargs.get('id2label')
        if self.id2label is None and hasattr(self.preprocessor, 'id2label'):
            self.id2label = self.preprocessor.id2label


@PIPELINES.register_module(
    Tasks.named_entity_recognition,
    module_name=Pipelines.named_entity_recognition_thai)
class NamedEntityRecognitionThaiPipeline(NamedEntityRecognitionPipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 **kwargs):
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        if preprocessor is None:
            self.preprocessor = NERPreprocessorThai(
                self.model.model_dir,
                sequence_length=kwargs.pop('sequence_length', 512))


@PIPELINES.register_module(
    Tasks.named_entity_recognition,
    module_name=Pipelines.named_entity_recognition_viet)
class NamedEntityRecognitionVietPipeline(NamedEntityRecognitionPipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 **kwargs):
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        if preprocessor is None:
            self.preprocessor = NERPreprocessorViet(
                self.model.model_dir,
                sequence_length=kwargs.pop('sequence_length', 512))
