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
        """Use `model` and `preprocessor` to create a nlp pair sequence classification pipeline for prediction.

        Args:
            model (str or Model): Supply either a local model dir which supported the sequence classification task,
            or a model id from the model hub, or a torch model instance.
            preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for
            the model if supplied.
            first_sequence: The key to read the first sentence in.
            second_sequence: The key to read the second sentence in.
            sequence_length: Max sequence length in the user's custom scenario. 512 will be used as a default value.

            NOTE: Inputs of type 'tuple' or 'list' are also supported. In this scenario, the 'first_sequence' and
            'second_sequence' param will have no effect.

            Example:
            >>> from modelscope.pipelines import pipeline
            >>> pipeline_ins = pipeline(task='nli', model='damo/nlp_structbert_nli_chinese-base')
            >>> sentence1 = '四川商务职业学院和四川财经职业学院哪个好？'
            >>> sentence2 = '四川商务职业学院商务管理在哪个校区？'
            >>> print(pipeline_ins((sentence1, sentence2)))
            >>> # Or use the dict input:
            >>> print(pipeline_ins({'first_sequence': sentence1, 'second_sequence': sentence2}))

            To view other examples plese check the tests/pipelines/test_nli.py.
        """
        if preprocessor is None:
            preprocessor = PairSentenceClassificationPreprocessor(
                model.model_dir if isinstance(model, Model) else model,
                first_sequence=first_sequence,
                second_sequence=second_sequence,
                sequence_length=kwargs.pop('sequence_length', 512))
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
