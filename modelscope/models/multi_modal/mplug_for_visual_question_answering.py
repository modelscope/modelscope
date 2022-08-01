from typing import Dict

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.base import Tensor
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks

__all__ = ['MPlugForVisualQuestionAnswering']


@MODELS.register_module(
    Tasks.visual_question_answering, module_name=Models.mplug)
class MPlugForVisualQuestionAnswering(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the mplug model from the `model_dir` path.
        Args:
            model_dir (str): the model path.
        """

        super().__init__(model_dir, *args, **kwargs)
        from modelscope.models.multi_modal.mplug import MPlugForVisualQuestionAnswering
        self.model = MPlugForVisualQuestionAnswering.from_pretrained(model_dir)
        self.tokenizer = self.model.tokenizer

    def train(self):
        return self.model.train()

    def eval(self):
        return self.model.eval()

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """return the result by the model

        Args:
            input (Dict[str, Tensor]): the preprocessed data

        Returns:
            Dict[str, Tensor]: results
                Example:
                    {
                        'predictions': Tensor([[1377, 4959, 2785, 6392...])]),
                    }
        """

        return self.model(**input)[0]
