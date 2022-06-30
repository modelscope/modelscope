from ...metainfo import Models
from ...utils.constant import Tasks
from ..builder import MODELS
from .sbert_for_sequence_classification import \
    SbertForSequenceClassificationBase

__all__ = ['SbertForSentimentClassification']


@MODELS.register_module(
    Tasks.sentiment_classification, module_name=Models.structbert)
class SbertForSentimentClassification(SbertForSequenceClassificationBase):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the text generation model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(
            model_dir, *args, model_args={'num_labels': 2}, **kwargs)
        assert self.model.config.num_labels == 2
