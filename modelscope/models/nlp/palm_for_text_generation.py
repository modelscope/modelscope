from typing import Dict

from modelscope.utils.constant import Tasks
from ..base import Model, Tensor
from ..builder import MODELS

__all__ = ['PalmForTextGeneration']


@MODELS.register_module(Tasks.text_generation, module_name=r'palm')
class PalmForTextGeneration(Model):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the text generation model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
            model_cls (Optional[Any], optional): model loader, if None, use the
                default loader to load model weights, by default None.
        """
        from sofa import PalmTokenizer

        super().__init__(model_dir, *args, **kwargs)
        self.model_dir = model_dir

        from sofa.models.palm import PalmForConditionalGeneration, TextGenerator
        tokenizer = kwargs.pop('tokenizer',
                               PalmTokenizer.from_pretrained(model_dir))
        model = PalmForConditionalGeneration.from_pretrained(model_dir)
        self.generator = TextGenerator(model, tokenizer)

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """return the result by the model

        Args:
            input (Dict[str, Any]): the preprocessed data

        Returns:
            Dict[str, np.ndarray]: results
                Example:
                    {
                        'predictions': array([1]), # lable 0-negative 1-positive
                        'probabilities': array([[0.11491239, 0.8850876 ]], dtype=float32),
                        'logits': array([[-0.53860897,  1.5029076 ]], dtype=float32) # true value
                    }
        """

        encoder_inputs = [
            input['input_ids'], input['token_type_ids'],
            input['attention_mask']
        ]
        return self.generator(encoder_inputs)
