# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from transformers import (GenerationConfig, PreTrainedTokenizer,
                          StoppingCriteriaList)
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from modelscope.metainfo import Models
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from ... import MODELS
from .backbone import QWenModel, QWenPreTrainedModel
from .qwen_generation_utils import (BatchTokensType, HistoryType,
                                    StopWordsLogitsProcessor, decode_tokens,
                                    get_batch, get_stop_words_ids,
                                    make_context, pad_batch, switch,
                                    top_k_logits)

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer

logger = get_logger()


@MODELS.register_module(Tasks.text_generation, module_name=Models.qwen_7b)
@MODELS.register_module(Tasks.chat, module_name=Models.qwen_7b)
class QWenForTextGeneration(QWenPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r'h\.\d+\.attn\.rotary_emb\.inv_freq']
    _keys_to_ignore_on_load_unexpected = [r'h\.\d+\.attn\.masked_bias']

    def __init__(self, config):
        super().__init__(config)
        self.transformer = QWenModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        assert not (config.bf16 and config.fp16
                    ), 'In config, bf16 and fp16 cannot both be true'
        if config.bf16:
            self.transformer.bfloat16()
            self.lm_head.bfloat16()
        if config.fp16:
            self.transformer.half()
            self.lm_head.half()
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past_key_values=None,
                                      inputs_embeds=None,
                                      **kwargs):
        token_type_ids = kwargs.get('token_type_ids', None)
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get('attention_mask', None)
        position_ids = kwargs.get('position_ids', None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
        })
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits, ) + transformer_outputs[1:]
            return ((loss, ) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(past_key_values: Tuple[Tuple[torch.Tensor]],
                       beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:

        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past)
            for layer_past in past_key_values)

    def chat(
        self,
        tokenizer: PreTrainedTokenizer,
        query: str,
        history: Optional[HistoryType],
        system: str = 'You are a helpful assistant.',
        append_history: bool = True,
    ) -> Tuple[str, HistoryType]:

        if history is None:
            history = []

        raw_text, context_tokens = make_context(
            tokenizer,
            query,
            history=history,
            system=system,
            max_window_size=6144,
            chat_format=self.generation_config.chat_format)

        stop_words_ids = get_stop_words_ids(self.generation_config.chat_format,
                                            tokenizer)
        input_ids = torch.tensor([context_tokens]).to(self.device)

        outputs = self.generate(
            input_ids,
            stop_words_ids=stop_words_ids,
            return_dict_in_generate=False,
        )

        response = decode_tokens(
            outputs[0],
            tokenizer,
            raw_text_len=len(raw_text),
            context_length=len(context_tokens),
            chat_format=self.generation_config.chat_format,
            verbose=False,
        )

        if append_history:
            history.append((query, response))

        return {OutputKeys.RESPONSE: response, OutputKeys.HISTORY: history}

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor],
                                                    List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        streamer: Optional['BaseStreamer'] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # Process stop_words_ids
        stop_words_ids = kwargs.pop('stop_words_ids', None)
        if stop_words_ids is None and generation_config is not None:
            stop_words_ids = getattr(generation_config, 'stop_words_ids', None)
        if stop_words_ids is None:
            stop_words_ids = getattr(self.generation_config, 'stop_words_ids',
                                     None)

        if stop_words_ids is not None:
            stop_words_logits_processor = StopWordsLogitsProcessor(
                stop_words_ids=stop_words_ids,
                eos_token_id=self.generation_config.eos_token_id)
            if logits_processor is None:
                logits_processor = LogitsProcessorList(
                    [stop_words_logits_processor])
            else:
                logits_processor.append(stop_words_logits_processor)

        return super().generate(
            inputs,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            streamer=streamer,
            **kwargs,
        )
