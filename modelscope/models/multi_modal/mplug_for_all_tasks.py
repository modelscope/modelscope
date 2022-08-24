from typing import Dict

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.base import Tensor
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks

__all__ = ['MPlugForAllTasks']


@MODELS.register_module(
    Tasks.visual_question_answering, module_name=Models.mplug)
@MODELS.register_module(Tasks.image_captioning, module_name=Models.mplug)
class MPlugForAllTasks(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the mplug model from the `model_dir` path.
        Args:
            model_dir (str): the model path.
        """

        super().__init__(model_dir, *args, **kwargs)
        from modelscope.models.multi_modal.mplug import MPlug
        self.model = MPlug.from_pretrained(model_dir)
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

        topk_ids, _ = self.model(**input)
        replace_tokens_bert = (('[unused0]', ''), ('[PAD]', ''),
                               ('[unused1]', ''), (r' +', ' '), ('[SEP]', ''),
                               ('[unused2]', ''), ('[CLS]', ''), ('[UNK]', ''))

        pred_string = self.tokenizer.decode(topk_ids[0][0])
        for _old, _new in replace_tokens_bert:
            pred_string = pred_string.replace(_old, _new)
        pred_string = pred_string.strip()
        return pred_string
