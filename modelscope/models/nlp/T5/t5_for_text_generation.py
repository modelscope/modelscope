from typing import Optional, Tuple

import torch

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Tasks
from .modeling_t5 import T5Config
from .modeling_t5 import T5ForConditionalGeneration as T5ForGeneration


@MODELS.register_module(
    group_key=Tasks.text2text_generation,
    module_name=Models.T5,
)
class T5ForConditionalGeneration(TorchModel):

    def __init__(self, model_dir=None, *args, **kwargs):
        """initialize the text generation model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
            model_cls (Optional[Any], optional): model loader, if None, use the
                default loader to load model weights, by default None.
        """
        super().__init__(model_dir, *args, **kwargs)
        self.model = T5ForGeneration.from_pretrained(model_dir)
        self.generate = self.model.generate
        self.config = self.model.config

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                decoder_attention_mask: Optional[torch.BoolTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                decoder_head_mask: Optional[torch.FloatTensor] = None,
                cross_attn_head_mask: Optional[torch.Tensor] = None,
                encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs):
        return self.model.forward(
            self, input_ids, attention_mask, decoder_input_ids,
            decoder_attention_mask, head_mask, decoder_head_mask,
            cross_attn_head_mask, encoder_outputs, past_key_values,
            inputs_embeds, decoder_inputs_embeds, labels, use_cache,
            output_attentions, output_hidden_states, return_dict, **kwargs)
