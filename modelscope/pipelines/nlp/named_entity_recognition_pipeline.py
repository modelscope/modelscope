# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Optional, Union

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.nlp import TokenClassificationPipeline
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import ModelFile, Tasks

__all__ = ['NamedEntityRecognitionPipeline']


@PIPELINES.register_module(
    Tasks.named_entity_recognition,
    module_name=Pipelines.named_entity_recognition)
@PIPELINES.register_module(
    Tasks.named_entity_recognition,
    module_name=Pipelines.named_entity_recognition_thai)
@PIPELINES.register_module(
    Tasks.named_entity_recognition,
    module_name=Pipelines.named_entity_recognition_viet)
class NamedEntityRecognitionPipeline(TokenClassificationPipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 config_file: str = None,
                 device: str = 'gpu',
                 auto_collate=True,
                 sequence_length=512,
                 **kwargs):
        """Use `model` and `preprocessor` to create a nlp NER pipeline for prediction

        Args:
            model (str or Model): Supply either a local model dir which supported NER task, or a
            model id from the model hub, or a torch model instance.
            preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for
                the model if supplied.
            kwargs (dict, `optional`):
                Extra kwargs passed into the preprocessor's constructor.

        Examples:
            >>> from modelscope.pipelines import pipeline
            >>> pipeline_ins = pipeline(task='named-entity-recognition',
            >>>        model='damo/nlp_raner_named-entity-recognition_chinese-base-news')
            >>> input = '这与温岭市新河镇的一个神秘的传说有关。'
            >>> print(pipeline_ins(input))

            To view other examples plese check the tests/pipelines/test_named_entity_recognition.py.
        """
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate)

        assert isinstance(self.model, Model), \
            f'please check whether model config exists in {ModelFile.CONFIGURATION}'

        if preprocessor is None:
            self.preprocessor = Preprocessor.from_pretrained(
                self.model.model_dir,
                sequence_length=sequence_length,
                **kwargs)
        self.model.eval()
        assert hasattr(self.preprocessor, 'id2label')
        self.id2label = self.preprocessor.id2label
