from ...metainfo import Models
from ...utils.constant import Tasks
from ..builder import MODELS
from .sbert_for_sequence_classification import \
    SbertForSequenceClassificationBase

__all__ = ['SbertForNLI']


@MODELS.register_module(Tasks.nli, module_name=Models.structbert)
class SbertForNLI(SbertForSequenceClassificationBase):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the text generation model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
            model_cls (Optional[Any], optional): model loader, if None, use the
                default loader to load model weights, by default None.
        """
        super().__init__(
            model_dir, *args, model_args={'num_labels': 3}, **kwargs)
        assert self.model.config.num_labels == 3
