from typing import Dict

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Tasks

__all__ = ['GPT3ForTextGeneration']


@MODELS.register_module(Tasks.text_generation, module_name=Models.gpt3)
class GPT3ForTextGeneration(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the text generation model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)

        from modelscope.models.nlp.gpt3 import GPT3Model
        from transformers import BertTokenizer

        self.model = GPT3Model.from_pretrained(model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """return the result by the model

        Args:
            input (Dict[str, Tensor]): the preprocessed data

        Returns:
            Dict[str, Tensor]: results
                Example:
                    {
                        'logits': Tensor([[0.54, 0.32...])]), # logits
                    }
        """
        return self.model(**input)

    def generate(self, input: Dict[str, Tensor]) -> Dict[str, str]:
        assert 'input_ids' in input, "generate function must accept 'input_ids' key"
        input_ids = input['input_ids']
        if 'attention_mask' in input:
            attention_mask = input['attention_mask']
            input_ids = input_ids[0][attention_mask[0].nonzero()] \
                .squeeze().unsqueeze(0)
        # remove sep token at the end of tokenizer output
        input_ids = input_ids[:, :-1]

        gen_params = dict()
        gen_params['inputs'] = input_ids
        gen_params['do_sample'] = input.pop('do_sample', True)
        gen_params['max_length'] = input.pop('max_length', 128)
        gen_params['top_k'] = input.pop('top_k', 10)
        gen_params['top_p'] = input.pop('top_p', None)
        sample_output = self.model.generate(**gen_params)
        return {
            OutputKeys.TEXT:
            self.tokenizer.decode(sample_output[0], skip_special_tokens=True)
        }
