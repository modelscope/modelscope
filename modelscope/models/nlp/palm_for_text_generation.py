from typing import Dict

from modelscope.metainfo import Models
from modelscope.utils.constant import Tasks
from ..base import Model, Tensor
from ..builder import MODELS

__all__ = ['PalmForTextGeneration']


@MODELS.register_module(Tasks.text_generation, module_name=Models.palm)
class PalmForTextGeneration(Model):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the text generation model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
            model_cls (Optional[Any], optional): model loader, if None, use the
                default loader to load model weights, by default None.
        """
        super().__init__(model_dir, *args, **kwargs)
        self.model_dir = model_dir

        from sofa.models.palm_v2 import PalmForConditionalGeneration, Translator
        model = PalmForConditionalGeneration.from_pretrained(model_dir)
        self.tokenizer = model.tokenizer
        self.generator = Translator(model)

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """return the result by the model

        Args:
            input (Dict[str, Tensor]): the preprocessed data

        Returns:
            Dict[str, Tensor]: results
                Example:
                    {
                        'predictions': Tensor([[1377, 4959, 2785, 6392...])]), # tokens need to be decode by tokenizer
                    }
        """

        return self.generator(**input)
