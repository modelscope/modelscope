from typing import Union

from modelscope.models.base import Model
from ...metainfo import Pipelines
from ...preprocessors import (PairSentenceClassificationPreprocessor,
                              Preprocessor)
from ...utils.constant import Tasks
from ..builder import PIPELINES
from .sequence_classification_pipeline_base import \
    SequenceClassificationPipelineBase

__all__ = ['PairSentenceClassificationPipeline']


@PIPELINES.register_module(Tasks.nli, module_name=Pipelines.nli)
@PIPELINES.register_module(
    Tasks.sentence_similarity, module_name=Pipelines.sentence_similarity)
class PairSentenceClassificationPipeline(SequenceClassificationPipelineBase):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Preprocessor = None,
                 first_sequence='first_sequence',
                 second_sequence='second_sequence',
                 **kwargs):
        """use `model` and `preprocessor` to create a nlp pair sentence classification pipeline for prediction

        Args:
            model (Model): a model instance
            preprocessor (Preprocessor): a preprocessor instance
        """
        if preprocessor is None:
            preprocessor = PairSentenceClassificationPreprocessor(
                model.model_dir if isinstance(model, Model) else model,
                first_sequence=first_sequence,
                second_sequence=second_sequence,
                sequence_length=kwargs.pop('sequence_length', 512))
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
