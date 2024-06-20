# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from contextlib import contextmanager
from threading import Lock
from typing import Any, Callable, Dict, Generator, Iterator, List, Tuple, Union

import json
import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from modelscope import (AutoModelForCausalLM, AutoTokenizer, Pipeline,
                        snapshot_download)
from modelscope.hub.file_download import model_file_download
from modelscope.models.base import Model
from modelscope.models.nlp import ChatGLM2Tokenizer, Llama2Tokenizer
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.util import is_model, is_official_hub_path
from modelscope.utils.config import Config
from modelscope.utils.constant import Frameworks, Invoke, ModelFile, Tasks
from modelscope.utils.device import create_device, device_placement
from modelscope.utils.logger import get_logger
from modelscope.utils.model_type_helper import ModelTypeHelper
from modelscope.utils.streaming_output import (PipelineStreamingOutputMixin,
                                               StreamingOutputMixin,
                                               add_stream_generate)

logger = get_logger()

SWIFT_MODEL_ID_MAPPING = {}


class LLMAdapterRegistry:

    llm_format_map = {'qwen': [None, None, None]}

    @classmethod
    def _add_to_map(cls, model_type: str, value_index: int = 0, member=None):
        assert model_type or ModelTypeHelper.current_model_type
        if model_type is None:
            model_type = ModelTypeHelper.current_model_type
        if model_type not in cls.llm_format_map:
            cls.llm_format_map[model_type] = [None, None, None]
        assert cls.llm_format_map[model_type][value_index] is None
        cls.llm_format_map[model_type][value_index] = member
        return member

    @classmethod
    def _wrapper(cls, model_type: str, value_index: int = 0, member=None):
        if member is not None:
            return cls._add_to_map(model_type, value_index, member)

        def _register(member):
            return cls._add_to_map(model_type, value_index, member)

        return _register

    @classmethod
    def register_format_messages(cls, model_type: str = None, function=None):
        return cls._wrapper(model_type, 0, function)

    @classmethod
    def register_format_output(cls, model_type: str = None, function=None):
        return cls._wrapper(model_type, 1, function)

    @classmethod
    def register_tokenizer(cls, model_type: str = None, tokenizer_class=None):
        return cls._wrapper(model_type, 2, tokenizer_class)

    @classmethod
    def contains(cls, model_name: str) -> bool:
        return model_name in cls.llm_format_map

    @classmethod
    def get(cls, model_name: str) -> bool:
        return cls.llm_format_map[model_name]


@PIPELINES.register_module(Tasks.chat, module_name='llm')
@PIPELINES.register_module(Tasks.text_generation, module_name='llm')
class LLMPipeline(Pipeline, PipelineStreamingOutputMixin):

    def initiate_single_model(self, model):
        from swift import Swift

        if isinstance(model, str):
            logger.info(f'initiate model from {model}')
        if self._is_swift_model(model):
            if self.llm_framework is not None:
                logger.warning(
                    f'Cannot using swift with llm_framework, ignoring {self.llm_framework}.'
                )

            base_model = self.cfg.safe_get('adapter_cfg.model_id_or_path')
            assert base_model is not None, 'Cannot get adapter_cfg.model_id_or_path from configuration.json file.'
            revision = self.cfg.safe_get('adapter_cfg.model_revision',
                                         'master')
            base_model = Model.from_pretrained(
                base_model,
                revision,
                invoked_by=Invoke.PIPELINE,
                device_map=self.device_map,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True)
            swift_model = Swift.from_pretrained(base_model, model_id=model)
            return swift_model

        if isinstance(model, str) and is_official_hub_path(model):
            logger.info(f'initiate model from location {model}.')
            if self.llm_framework:
                model_dir = model if os.path.exists(
                    model) else snapshot_download(model)
                try:
                    model = self._wrap_infer_framework(model_dir,
                                                       self.llm_framework)
                    logger.info(f'initiate model with {self.llm_framework}.')
                    return model
                except Exception as e:
                    logger.warning(
                        f'Cannot using llm_framework with {model}, '
                        f'ignoring llm_framework={self.llm_framework} : {e}')
                    self.llm_framework = None
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

    def _is_swift_model(self, model: Union[str, Any]) -> bool:
        if not isinstance(model, str):
            return False
        if os.path.exists(model):
            cfg_file = os.path.join(model, ModelFile.CONFIGURATION)
        else:
            try:
                cfg_file = model_file_download(model, ModelFile.CONFIGURATION)
            except Exception:
                return False

        self.cfg = Config.from_file(cfg_file)
        return self.cfg.safe_get('adapter_cfg.tuner_backend') == 'swift'

    def _wrap_infer_framework(self, model_dir, framework='vllm'):
        from modelscope.pipelines.accelerate.base import InferFramework
        return InferFramework.from_pretrained(model_dir, framework)

    def __init__(self,
                 format_messages: Union[Callable, str] = None,
                 format_output: Callable = None,
                 tokenizer: PreTrainedTokenizer = None,
                 llm_framework: str = None,
                 *args,
                 **kwargs):
        self.device_map = kwargs.pop('device_map', None)
        self.llm_framework = llm_framework
        # TODO: qwen-int4 need 'cuda'/'auto' device_map.
        if not self.device_map and 'qwen' in kwargs['model'].lower():
            self.device_map = 'cuda'
        self.torch_dtype = kwargs.pop('torch_dtype', None)
        self.ignore_file_pattern = kwargs.pop('ignore_file_pattern', None)

        if llm_framework == 'swift':
            self._init_swift(kwargs['model'], kwargs.get('device', 'gpu'))
            return
        with self._temp_configuration_file(kwargs):
            super().__init__(*args, **kwargs)
        if isinstance(self.model, PreTrainedModel):
            self.model = add_stream_generate(self.model)

        tokenizer_class = None
        if isinstance(format_messages, str):
            assert LLMAdapterRegistry.contains(format_messages), \
                f'Can not find function for `{format_messages}`!'
            format_messages, format_output, tokenizer_class = \
                LLMAdapterRegistry.get(format_messages)

        if format_messages is None:
            model_type = ModelTypeHelper.get(self.model.model_dir, split='-')
            if LLMAdapterRegistry.contains(model_type):
                format_messages, format_output, tokenizer_class = \
                    LLMAdapterRegistry.get(model_type)

        if format_messages is not None:
            self.format_messages = format_messages
        if format_output is not None:
            self.format_output = format_output
        self.tokenizer = self._get_tokenizer(
            tokenizer_class) if tokenizer is None else tokenizer

    def _init_swift(self, model_id, device) -> None:
        from swift.llm import prepare_model_template
        from swift.llm.utils import MODEL_MAPPING, InferArguments

        global SWIFT_MODEL_ID_MAPPING
        if not SWIFT_MODEL_ID_MAPPING:
            SWIFT_MODEL_ID_MAPPING = {
                v['model_id_or_path']: k
                for k, v in MODEL_MAPPING.items()
            }

        def format_messages(messages: Dict[str, List[Dict[str, str]]],
                            tokenizer: PreTrainedTokenizer,
                            **kwargs) -> Dict[str, torch.Tensor]:
            inputs, _ = self.template.encode(get_example(messages))
            inputs.pop('labels', None)
            if 'input_ids' in inputs:
                input_ids = torch.tensor(inputs['input_ids'])[None]
                inputs['input_ids'] = input_ids
                token_len = input_ids.shape[1]
            if 'inputs_embeds' in inputs:
                inputs_embeds = inputs['inputs_embeds'][None]
                inputs['inputs_embeds'] = inputs_embeds
                token_len = inputs_embeds.shape[1]
            inputs['attention_mask'] = torch.ones(token_len)[None]
            if 'token_type_ids' in inputs:
                inputs['token_type_ids'] = torch.tensor(
                    inputs['token_type_ids'])[None]
            return inputs

        def get_example(
                messages: Dict[str, List[Dict[str, str]]]) -> Dict[str, str]:
            messages = messages['messages']
            assert len(messages) > 0, 'messages cannot be empty!'
            system = None
            if messages[0]['role'] == 'system':
                system = messages[0]['content']
                messages = messages[1:]
            assert len(messages) % 2 == 1, 'Unsupported messages format!'
            contents = [message['content'] for message in messages]
            prompt = contents[-1]
            history = list(zip(contents[::2], contents[1::2]))
            return dict(system=system, prompt=prompt, history=history)

        assert model_id in SWIFT_MODEL_ID_MAPPING, 'Swift framework does not support current model!'
        args = InferArguments(model_type=SWIFT_MODEL_ID_MAPPING[model_id])
        model, template = prepare_model_template(
            args, device_map=self.device_map)
        self.model = add_stream_generate(model)
        template.model = self.model
        self.template = template
        self.tokenizer = template.tokenizer
        self.format_messages = format_messages

        self.has_multiple_models = False
        self.framework = Frameworks.torch
        self.device_name = device
        self.device = create_device(device)
        self._model_prepare = False
        self._model_prepare_lock = Lock()
        self._auto_collate = True
        self._compile = False

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

        preprocess_params['is_messages'] = postprocess_params['is_messages'] \
            = isinstance(inputs, dict) and 'messages' in inputs
        tokens = self.preprocess(inputs, **preprocess_params)

        if self.llm_framework in (None, 'swift'):
            # pytorch model
            if hasattr(self.model, 'generate'):
                outputs = self.model.generate(**tokens, **forward_params)
            elif hasattr(self.model, 'model') and hasattr(
                    self.model.model, 'generate'):
                outputs = self.model.model.generate(**tokens, **forward_params)
            else:
                raise ValueError('model does not support `generate`!')
        else:
            tokens = [list(tokens['inputs'].flatten().numpy())]
            outputs = self.model(tokens, **forward_params)[0]

        if self.llm_framework is None:
            # pytorch model
            outputs = outputs.tolist()[0][len(tokens['inputs'][0]):]
        response = self.postprocess(outputs, **postprocess_params)
        return response

    def stream_generate(self, inputs: Union[Input, List[Input]], *args,
                        **kwargs) -> Generator:
        assert isinstance(self.model, StreamingOutputMixin
                          ), 'pipeline.model must be StreamingOutputMixin!'
        if (self.model or (self.has_multiple_models and self.models[0])):
            if not self._model_prepare:
                self.prepare_model()

        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(
            **kwargs)
        preprocess_params['is_messages'] = postprocess_params['is_messages'] \
            = isinstance(inputs, dict) and 'messages' in inputs

        if isinstance(inputs, list):
            model_input_list = [
                self._preprocess_with_check(i, preprocess_params)
                for i in inputs
            ]
            output = []
            for ele in model_input_list:
                output.append(
                    self._stream_single(ele, forward_params,
                                        postprocess_params))
        else:
            model_input = self._preprocess_with_check(inputs,
                                                      preprocess_params)
            output = self._stream_single(model_input, forward_params,
                                         postprocess_params)
        return output

    def _stream_single(self, model_input: Dict[str, Any],
                       forward_params: Dict[str, Any],
                       postprocess_params: Dict[str, Any]) -> Generator:

        with device_placement(self.framework, self.device_name):
            if self.framework == Frameworks.torch:
                with torch.no_grad():
                    if self._auto_collate:
                        model_input = self._collate_fn(model_input)
                    stream = self.model.stream_generate(
                        **model_input, **forward_params)
            else:
                stream = self.model.stream_generate(**model_input,
                                                    **forward_params)

            for out in stream:
                out = out.tolist()[0][len(model_input['inputs'][0]):]
                out = self.postprocess(out, **postprocess_params)
                self._check_output(out)
                yield out

    def preprocess(self, inputs: Union[str, Dict], **kwargs):
        is_messages = kwargs.pop('is_messages')
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
        elif hasattr(self.model, 'llm_framework'):
            device = 'cpu'
        else:
            raise ValueError('model does not have `device` attribute!')
        return {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in tokens.items()
        }

    def postprocess(self, outputs, **kwargs):
        is_messages = kwargs.pop('is_messages')
        if not isinstance(outputs, str):
            shape_type = (torch.Tensor, np.ndarray)
            if isinstance(outputs, shape_type) and len(outputs.shape) > 1:
                outputs = outputs[0]
            response = self.tokenizer.decode(
                outputs, skip_special_tokens=True, **kwargs)
        else:
            response = outputs
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


@LLMAdapterRegistry.register_format_messages('chatglm2')
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


@LLMAdapterRegistry.register_format_output('chatglm')
@LLMAdapterRegistry.register_format_output('chatglm2')
def chatglm2_format_output(response, **kwargs):
    response = response.strip()
    response = response.replace('[[训练时间]]', '2023年')
    messages = {'role': 'assistant', 'content': response}
    outputs = {
        'message': messages,
    }
    return outputs


@LLMAdapterRegistry.register_format_messages('llama')
@LLMAdapterRegistry.register_format_messages('llama2')
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


@LLMAdapterRegistry.register_format_messages('baichuan')
@LLMAdapterRegistry.register_format_messages('baichuan2')
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


@LLMAdapterRegistry.register_format_messages('wizardlm')
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


@LLMAdapterRegistry.register_format_messages('wizardcode')
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


@LLMAdapterRegistry.register_format_messages('chatglm')
def chatglm3_format_messages(messages, tokenizer, **kwargs):
    messages = messages['messages']
    query, history = messages[-1]['content'], messages[:-1]
    inputs = tokenizer.build_chat_input(query, history=history)
    eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.get_command('<|user|>'),
        tokenizer.get_command('<|observation|>')
    ]
    inputs['eos_token_id'] = eos_token_id
    return inputs


@LLMAdapterRegistry.register_format_messages('qwen2')
def qwen2_format_messages(messages, tokenizer, **kwargs):
    messages = messages['messages']
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    return tokenizer([text], return_tensors='pt')


LLMAdapterRegistry.register_tokenizer('chatglm2', ChatGLM2Tokenizer)
LLMAdapterRegistry.register_tokenizer('llama', Llama2Tokenizer)
LLMAdapterRegistry.register_tokenizer('llama2', Llama2Tokenizer)
