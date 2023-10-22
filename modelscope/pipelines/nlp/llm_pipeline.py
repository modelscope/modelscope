# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Tuple, Union

import json
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from modelscope import (AutoModelForCausalLM, AutoTokenizer, Pipeline,
                        snapshot_download)
from modelscope.models.base import Model
from modelscope.models.nlp import ChatGLM2Tokenizer, Llama2Tokenizer
from modelscope.outputs import OutputKeys
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.util import is_model, is_official_hub_path
from modelscope.utils.constant import Invoke, ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(Tasks.chat, module_name='llm')
@PIPELINES.register_module(Tasks.text_generation, module_name='llm')
class LLMPipeline(Pipeline):

    def initiate_single_model(self, model):
        if isinstance(model, str):
            logger.info(f'initiate model from {model}')
        if isinstance(model, str) and is_official_hub_path(model):
            logger.info(f'initiate model from location {model}.')
            if is_model(model):
                return Model.from_pretrained(
                    model,
                    invoked_by=Invoke.PIPELINE,
                    device_map=self.device_map,
                    torch_dtype=self.torch_dtype,
                    ignore_file_pattern=self.ignore_file_pattern)
            else:
                model_dir = model if os.path.exists(
                    model) else snapshot_download(model)
                # TODO: Temporary use of AutoModelForCausalLM
                # Need to be updated into a universal solution
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    device_map=self.device_map,
                    trust_remote_code=True)
                model.model_dir = model_dir
                return model
        else:
            return model

    def __init__(self,
                 format_messages: Union[Callable, str] = None,
                 format_output: Callable = None,
                 tokenizer: PreTrainedTokenizer = None,
                 *args,
                 **kwargs):
        self.device_map = kwargs.pop('device_map', None)
        # TODO: qwen-int4 need 'cuda'/'auto' device_map.
        if not self.device_map and 'qwen' in kwargs['model'].lower():
            self.device_map = 'cuda'
        self.torch_dtype = kwargs.pop('torch_dtype', None)
        self.ignore_file_pattern = kwargs.pop('ignore_file_pattern', None)
        with self._temp_configuration_file(kwargs):
            super().__init__(*args, **kwargs)

        tokenizer_class = None
        if isinstance(format_messages, str):
            assert format_messages in LLM_FORMAT_MAP, \
                f'Can not find function for `{format_messages}`!'
            format_messages, format_output, tokenizer_class = LLM_FORMAT_MAP[
                format_messages]

        if format_messages is None:
            model_type = self.cfg.safe_get('model.type',
                                           '').lower().split('-')[0]
            if model_type in LLM_FORMAT_MAP:
                format_messages, format_output, tokenizer_class = LLM_FORMAT_MAP[
                    model_type]

        if format_messages is not None:
            self.format_messages = format_messages
        if format_output is not None:
            self.format_output = format_output
        self.tokenizer = self._get_tokenizer(
            tokenizer_class) if tokenizer is None else tokenizer

        print(
            f'\n>>self.tokenizer: tokenizer: {self.tokenizer}, >tokenizer_class: {tokenizer_class}\n'
        )

    @contextmanager
    def _temp_configuration_file(self, kwargs: Dict[str, Any]):
        kwargs['model'] = model = self.initiate_single_model(kwargs['model'])
        model_dir = model if isinstance(model, str) else model.model_dir
        configuration_path = os.path.join(model_dir, ModelFile.CONFIGURATION)
        if os.path.exists(configuration_path):
            yield
        else:
            with open(configuration_path, 'w') as f:
                json.dump({'framework': 'pytorch', 'task': 'chat'}, f)
            yield
            os.remove(configuration_path)

    def _process_single(self, inputs, *args, **kwargs) -> Dict[str, Any]:
        preprocess_params = kwargs.get('preprocess_params', {})
        forward_params = kwargs.get('forward_params', {})
        postprocess_params = kwargs.get('postprocess_params', {})

        is_messages = isinstance(inputs, dict) and 'messages' in inputs
        tokens = self.preprocess(inputs, is_messages, **preprocess_params)
        print(f'\n>> info in _process_single: '
              f'\n>tokens: {tokens}, '
              f'\n>is_messages: {is_messages}, '
              f'\n>forward_params: {forward_params}, '
              f'\n>preprocess_params: {preprocess_params}, '
              f'\n>postprocess_params: {postprocess_params}, '
              f'\n>model has generate: {hasattr(self.model, "generate")}, '
              f'\n>model has model: {hasattr(self.model, "model")}')

        if hasattr(self.model, 'generate'):
            # self.generate: <bound method GenerationMixin.generate of ChatGLM2ForConditionalGeneration>
            # Func ref: transformers.generation.utils.GenerationMixin.generate
            outputs = self.model.generate(**tokens, **forward_params)
            print(f'>>>self.model.generate: {self.model.generate}')

            outputs_new: CausalLMOutputWithPast = self.model(tokens['inputs'])
            print(
                f'\n\n>>outputs_new in _process_single for llm_pipe model call: '
                # f'\n>data: {outputs_new}'
                f'\n>logits: {outputs_new.logits}'
                f'\n>logits shape: {outputs_new.logits.shape}'
                f'\n>type: {type(outputs_new)}\n\n')

            outputs_logits = outputs_new[0]
            print('>>outputs_logits shape: ', outputs_logits.shape)
            multi_logits = F.log_softmax(outputs_logits, dim=-1).cpu()
            print('>>multi_logits shape: ', multi_logits.shape)
            print(multi_logits[0])

        elif hasattr(self.model, 'model') and hasattr(self.model.model,
                                                      'generate'):
            outputs = self.model.model.generate(**tokens, **forward_params)
        else:
            raise ValueError('model does not support `generate`!')

        print(f'\n\n>>outputs in _process_single for llm_pipe: '
              f'\n>data: {outputs}'
              f'\n>shape: {outputs.shape}'
              f'\n>type: {type(outputs)}\n\n')

        # Get tokens of the generated continuation
        outputs = outputs.tolist()[0][len(tokens['inputs'][0]):]

        print(
            f'>>tokens of continuation in _process_single for llm_pipe: {outputs}'
        )

        response = self.postprocess(outputs, is_messages, **postprocess_params)

        print(f'>>response in _process_single for llm_pipe: {response}')

        print(f'\n>>self.model:\n {self.model}')

        return response

    def preprocess(self, inputs: Union[str, Dict], is_messages: bool,
                   **kwargs):
        if is_messages:
            tokens = self.format_messages(inputs, self.tokenizer, **kwargs)
        else:
            tokens = self.tokenizer(inputs, return_tensors='pt', **kwargs)

        tokens['inputs'] = tokens.pop('input_ids')

        if hasattr(self.model, 'device'):
            device = self.model.device
        elif hasattr(self.model, 'model') and hasattr(self.model.model,
                                                      'device'):
            device = self.model.model.device
        else:
            raise ValueError('model does not have `device` attribute!')
        return {k: v.to(device) for k, v in tokens.items()}

    def postprocess(self, outputs, is_messages: bool, **kwargs):

        response = self.tokenizer.decode(
            outputs, skip_special_tokens=True, **kwargs)
        if is_messages:
            response = self.format_output(response, **kwargs)
        else:
            response = {OutputKeys.TEXT: response}

        return response

    def _sanitize_parameters(self, **generate_parameter):
        """
        this method should sanitize the keyword args to preprocessor params,
        forward params and postprocess params on '__call__' or '_process_single' method
        considered to be a normal classmethod with default implementation / output

        Default Returns:
            Dict[str, str]:  preprocess_params = {}
            Dict[str, str]:  forward_params = {}
            Dict[str, str]:  postprocess_params = pipeline_parameters
        """
        return {}, generate_parameter, {}

    def _get_tokenizer(self, tokenizer_class=None):
        if isinstance(self.model, str):
            model_dir = self.model
        else:
            model_dir = self.model.model_dir
        if tokenizer_class is None:
            tokenizer_class = AutoTokenizer
        return tokenizer_class.from_pretrained(
            model_dir, trust_remote_code=True)

    @staticmethod
    def format_messages(messages: Dict[str, List[Dict[str, str]]],
                        tokenizer: PreTrainedTokenizer,
                        **kwargs) -> Dict[str, torch.Tensor]:
        # {"messages":[{"role": "system", "content": "You are a helpful assistant."}...]}
        tokens = []
        for role, content in LLMPipeline._message_iter(messages):
            tokens = LLMPipeline._concat_with_special_tokens(
                tokens, role, content, tokenizer)
        return {'input_ids': torch.tensor([tokens], dtype=torch.int64)}

    @staticmethod
    def format_output(response: str, **kwargs):
        response = response.strip()
        message = {'message': {'role': 'assistant', 'content': response}}
        return message

    @staticmethod
    def _message_iter(
            data: Dict[str, List[Dict[str,
                                      str]]]) -> Iterator[Tuple[str, str]]:
        for pair in data['messages']:
            yield pair['role'], pair['content']

    @staticmethod
    def _concat_with_special_tokens(
            ids: List[int], role: str, content: Union[str, List[Dict[str,
                                                                     str]]],
            tokenizer: PreTrainedTokenizer) -> List[int]:
        im_start = tokenizer.im_start_id
        im_end = tokenizer.im_end_id
        nl_token = tokenizer.encode('\n')
        role = tokenizer.encode(role.strip())
        content = LLMPipeline._encode(tokenizer, content)
        return LLMPipeline._concat(ids, im_start, role, nl_token, content,
                                   im_end, nl_token)

    @staticmethod
    def _encode(tokenizer: PreTrainedTokenizer,
                content: Union[str, List[Dict[str, str]]]):
        if isinstance(content, str):
            return tokenizer.encode(content.rstrip())
        encoded = []
        for pair in content:
            (modal, value), = pair.items()
            if modal == 'image':
                img_token_span = getattr(tokenizer, 'img_token_span', 256)
                img_start_id = tokenizer.img_start_id
                img_end_id = img_start_id + 1
                img_pad_id = img_start_id + 2
                list_int_url = list(bytes(value, encoding='utf-8'))
                assert len(
                    list_int_url) <= img_token_span, 'Image url is too long.'
                pad_ids = [img_pad_id] * (img_token_span - len(list_int_url))
                encoded = LLMPipeline._concat(encoded, img_start_id,
                                              list_int_url, pad_ids,
                                              img_end_id)
            else:  # text
                encoded.extend(tokenizer.encode(value))
        return encoded

    @staticmethod
    def _concat(ids: List[int], *args: Union[int, List[int]]) -> List[int]:
        for item in args:
            if isinstance(item, list):
                ids.extend(item)
            else:
                ids.append(item)
        return ids


def chatglm2_format_messages(messages, tokenizer, **kwargs):

    def build_chatglm2_prompt(messages, **kwargs):
        prompt = ''
        messages = messages['messages']
        # chatglm2 does not have system messages
        assert messages[0][
            'role'] == 'user', 'chatglm2 does not have system messages'

        for i in range(0, len(messages) - 1, 2):
            prompt += '[Round {}]\n\n问：{}\n\n答：{}\n\n'.format(
                i // 2 + 1, messages[i]['content'], messages[i + 1]['content'])
        prompt += '[Round {}]\n\n问：{}\n\n答：'.format(
            len(messages) // 2 + 1, messages[-1]['content'])
        return prompt

    prompt = build_chatglm2_prompt(messages, **kwargs)
    return tokenizer(prompt, return_token_type_ids=False, return_tensors='pt')


def chatglm2_format_output(response, **kwargs):
    response = response.strip()
    response = response.replace('[[训练时间]]', '2023年')
    messages = {'role': 'assistant', 'content': response}
    outputs = {
        'message': messages,
    }
    return outputs


def llama2_format_messages(messages, tokenizer, **kwargs):
    from transformers import BatchEncoding

    def build_llama2_prompt(messages, tokenizer, **kwargs):
        max_length = kwargs.get('max_length', 2048)
        default_system_message = 'you are a helpful assistant!'

        messages = messages['messages']
        # llama2 have system messages
        if messages[0]['role'] != 'system':
            messages = [{
                'role': 'system',
                'content': default_system_message
            }] + messages

        system = messages[0]['content']
        system_prompt = f'<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n'
        system_ids = tokenizer(system_prompt, return_tensors='pt').input_ids

        text = messages[-1]['content']
        text_prompt = f'{text.strip()} [/INST]'
        text_ids = tokenizer(text_prompt, return_tensors='pt').input_ids
        prompt_length = system_ids.shape[-1] + text_ids.shape[-1]
        if prompt_length > max_length:
            raise RuntimeError(
                f'prepend prompt length {prompt_length} is bigger than max_length {max_length}'
            )

        # history items
        history_prompt = ''
        history_ids_list = []
        for i in range(len(messages) - 2, 0, -2):
            user, assistant = messages[i]['content'], messages[i
                                                               + 1]['content']
            round_prompt = f'{user.strip()} [/INST] {assistant.strip()} </s><s>[INST] '
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
        return ''.join(prompt_list), torch.cat(prompt_ids_list, dim=-1)

    prompt, tokens = build_llama2_prompt(messages, tokenizer, **kwargs)
    return BatchEncoding({'input_ids': tokens})


def baichuan_format_messages(messages, tokenizer, **kwargs):
    from transformers import BatchEncoding

    def _parse_messages(messages, split_role='user'):
        system, rounds = '', []
        round = []
        for i, message in enumerate(messages):
            if message['role'] == 'system':
                assert i == 0, 'first message should be system message.'
                system = message['content']
                continue
            if message['role'] == split_role and round:
                rounds.append(round)
                round = []
            round.append(message)
        if round:
            rounds.append(round)
        return system, rounds

    messages = messages['messages']
    assistant_token_id = 196
    user_token_id = 195
    max_new_tokens = kwargs.get('max_new_tokens', None) or 2048
    model_max_length = 4096
    max_input_tokens = model_max_length - max_new_tokens
    system, rounds = _parse_messages(messages, split_role='user')
    system_tokens = tokenizer.encode(system)
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    for round in rounds[::-1]:
        round_tokens = []
        for message in round:
            if message['role'] == 'user':
                round_tokens.append(user_token_id)
            else:
                round_tokens.append(assistant_token_id)
            round_tokens.extend(tokenizer.encode(message['content']))
        if len(history_tokens) == 0 or len(history_tokens) + len(
                round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + history_tokens
    if messages[-1]['role'] != 'assistant':
        input_tokens.append(assistant_token_id)
    input_tokens = input_tokens[-max_input_tokens:]  # truncate left
    input_tokens = torch.LongTensor([input_tokens])
    return BatchEncoding({'input_ids': input_tokens})


def wizardlm_format_messages(messages, tokenizer, **kwargs):

    def build_wizardlm_prompt(messages, tokenizer, **kwargs):
        default_system_message = 'A chat between a curious user and an artificial intelligence assistant.'
        'The assistant gives helpful, detailed, and polite answers to the user\'s questions.'

        messages = messages['messages']
        # llama2 have system messages
        if messages[0]['role'] != 'system':
            messages = [{
                'role': 'system',
                'content': default_system_message
            }] + messages

        system_prompt = messages[0]['content']
        prompt_list = [system_prompt]
        for i, message in enumerate(messages[1:]):
            if message['role'] == 'user':
                user_prompt = message['content']
                prompt_list.append(f'USER: {user_prompt}')
            elif message['role'] == 'assistant':
                user_prompt = message['content']
                prompt_list.append(f'ASSISTANT: {user_prompt}</s>')
        prompts = ' '.join(prompt_list)
        return prompts

    prompts = build_wizardlm_prompt(messages, tokenizer, **kwargs)
    return tokenizer(prompts, return_token_type_ids=False, return_tensors='pt')


def wizardcode_format_messages(messages, tokenizer, **kwargs):
    messages = messages['messages']
    assert len(messages) == 2, 'wizard code only support two messages.'
    system, user = '', ''
    for i, message in enumerate(messages):
        if message['role'] == 'system':
            assert i == 0, 'first message should be system message.'
            system = message['content']
        if message['role'] == 'user':
            assert i == 1, 'second message should be user message.'
            user = message['content']

    prompt = system + '\n\n### Instruction:\n' + user + '\n\n### Response:'
    inputs = tokenizer(
        prompt,
        return_token_type_ids=False,
        padding=False,
        add_special_tokens=False,
        return_tensors='pt')
    return inputs


LLM_FORMAT_MAP = {
    'chatglm2':
    (chatglm2_format_messages, chatglm2_format_output, ChatGLM2Tokenizer),
    'qwen': (LLMPipeline.format_messages, LLMPipeline.format_output, None),
    'llama2': (llama2_format_messages, None, Llama2Tokenizer),
    'llama': (llama2_format_messages, None, Llama2Tokenizer),
    'baichuan': (baichuan_format_messages, None, None),
    'baichuan2': (baichuan_format_messages, None, None),
    'wizardlm': (wizardlm_format_messages, None, None),
    'wizardcode': (wizardcode_format_messages, None, None)
}
