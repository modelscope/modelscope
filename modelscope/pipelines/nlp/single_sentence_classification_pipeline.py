from typing import Union

from ...metainfo import Pipelines
from ...models import Model
from ...preprocessors import (Preprocessor,
                              SingleSentenceClassificationPreprocessor)
from ...utils.constant import Tasks
from ..builder import PIPELINES
from .sequence_classification_pipeline_base import \
    SequenceClassificationPipelineBase

__all__ = ['SingleSentenceClassificationPipeline']


@PIPELINES.register_module(
    Tasks.sentiment_classification,
    module_name=Pipelines.sentiment_classification)
class SingleSentenceClassificationPipeline(SequenceClassificationPipelineBase):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Preprocessor = None,
                 first_sequence='first_sequence',
                 **kwargs):
        """use `model` and `preprocessor` to create a nlp single sentence classification pipeline for prediction

        Args:
            model (Model): a model instance
            preprocessor (Preprocessor): a preprocessor instance
        """
        if preprocessor is None:
            preprocessor = SingleSentenceClassificationPreprocessor(
                model.model_dir if isinstance(model, Model) else model,
                first_sequence=first_sequence)
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
