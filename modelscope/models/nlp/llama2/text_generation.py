from modelscope.metainfo import Models
from modelscope.models.builder import MODELS
from modelscope.models.nlp.llama import \
    LlamaForTextGeneration as Llama2ForTextGeneration
from modelscope.utils.constant import Tasks
from ... import MODELS
from .backbone import Llama2Model, LlamaPreTrainedModel


def get_chat_prompt(system: str, text: str, history: List[Tuple[str, str]],
                    max_length: int, tokenizer):
    system_prompt = f'<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n'
    system_ids = tokenizer(system_prompt, return_tensors='pt').input_ids

    text_prompt = f'{text.strip()} [/INST]'
    text_ids = tokenizer(text_prompt, return_tensors='pt').input_ids

    prompt_length = system_ids.shape[-1] + text_ids.shape[-1]
    if prompt_length > max_length:
        raise RuntimeError(
            f'prepend prompt length {prompt_length} is bigger than max_length {max_length}'
        )

    history_prompt = ''
    history_ids_list = []
    # traverse history in reverse order
    for user, bot in history[::-1]:
        assert isinstance(user, str)
        assert isinstance(bot, str)
        round_prompt = f'{user.strip()} [/INST] {bot.strip()} </s><s>[INST] '
        round_ids = tokenizer(round_prompt, return_tensors='pt').input_ids
        if prompt_length + round_ids.shape[-1] > max_length:
            # excess history should not be appended to the prompt
            break
        else:
            history_prompt = round_prompt + history_prompt
            history_ids_list = [round_ids] + history_ids_list
            prompt_length += round_ids.shape[-1]

    prompt_list = [system_prompt, history_prompt, text_prompt]
    prompt_ids_list = [system_ids] + history_ids_list + [text_ids]

    return ''.join(prompt_list), torch.cat(prompt_ids_list, dim=1)


# This file is mainly copied from the llama code of transformers
@MODELS.register_module(Tasks.text_generation, module_name=Models.llama2)
class Llama2ForTextGeneration(LlamaPreTrainedModel):
    _tied_weights_keys = ['lm_head.weight']

    def __init__(self, config):
        super().__init__(config)
        self.model = Llama2Model(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.pretraining_tp, dim=0)
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits, ) + outputs[1:]
            return (loss, ) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past_key_values=None,
                                      attention_mask=None,
                                      inputs_embeds=None,
                                      **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}

        model_inputs.update({
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
        })
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past), )
        return reordered_past

    def chat(self, input: Dict, tokenizer) -> Dict:
        import copy
        gen_kwargs = copy.copy(input)
        if 'text' not in input:
            text: str = ''
        else:
            text: str = input['text']
            gen_kwargs.pop('text')

        if 'system' not in input:
            system: str = ''
        else:
            system: str = input['system']
            gen_kwargs.pop('system')

        if 'history' not in input:
            history = []
        else:
            history: List[Tuple] = copy.copy(input['history'])
            gen_kwargs.pop('history')

        if 'max_length' not in gen_kwargs:
            gen_kwargs['max_length'] = 4096

        prompt, prompt_ids = get_chat_prompt(
            system=system,
            text=text,
            history=history,
            max_length=gen_kwargs['max_length'],
            tokenizer=tokenizer)
        input_ids = prompt_ids.to(self.device)
        generate_ids = self.generate(input_ids, **gen_kwargs)
        # remove input tokens
        generate_ids = generate_ids[:, input_ids.shape[1]:]
        response = tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        response = response.strip()
        history.append((text, response))

        return {OutputKeys.RESPONSE: response, OutputKeys.HISTORY: history}

    @torch.no_grad()
    def generate(self,
                 inputs=None,
                 generation_config=None,
                 logits_processor=None,
                 stopping_criteria=None,
                 prefix_allowed_tokens_fn=None,
                 synced_gpus=None,
                 assistant_model=None,
                 streamer=None,
                 **kwargs):
        if inputs.device.type != self.device.type:
            inputs = inputs.to(self.device)
        return super().generate(inputs, generation_config, logits_processor,
                                stopping_criteria, prefix_allowed_tokens_fn,
                                synced_gpus, assistant_model, streamer,
                                **kwargs)
