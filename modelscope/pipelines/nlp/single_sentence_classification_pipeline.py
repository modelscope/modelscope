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
        """Use `model` and `preprocessor` to create a nlp single sequence classification pipeline for prediction.

        Args:
            model (str or Model): Supply either a local model dir which supported the sequence classification task,
            or a model id from the model hub, or a torch model instance.
            preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for
            the model if supplied.
            first_sequence: The key to read the first sentence in.
            sequence_length: Max sequence length in the user's custom scenario. 512 will be used as a default value.

            NOTE: Inputs of type 'str' are also supported. In this scenario, the 'first_sequence'
            param will have no effect.

            Example:
            >>> from modelscope.pipelines import pipeline
            >>> pipeline_ins = pipeline(task='sentiment-classification',
            >>>    model='damo/nlp_structbert_sentiment-classification_chinese-base')
            >>> sentence1 = '启动的时候很大声音，然后就会听到1.2秒的卡察的声音，类似齿轮摩擦的声音'
            >>> print(pipeline_ins(sentence1))
            >>> # Or use the dict input:
            >>> print(pipeline_ins({'first_sequence': sentence1}))

            To view other examples plese check the tests/pipelines/test_sentiment-classification.py.
        """
        if preprocessor is None:
            preprocessor = SingleSentenceClassificationPreprocessor(
                model.model_dir if isinstance(model, Model) else model,
                first_sequence=first_sequence,
                sequence_length=kwargs.pop('sequence_length', 512))
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
