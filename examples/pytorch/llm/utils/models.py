from typing import NamedTuple

import torch
from torch import dtype as Dtype

from modelscope import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, Model,
                        get_logger, read_config, snapshot_download)
from modelscope.models.nlp.chatglm2 import ChatGLM2Config, ChatGLM2Tokenizer

logger = get_logger()


def _add_special_token(tokenizer):
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = 2
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = 1
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    logger.info(f'bos_token_id: {tokenizer.bos_token_id}, '
                f'eos_token_id: {tokenizer.eos_token_id}, '
                f'pad_token_id: {tokenizer.pad_token_id}')


def get_baichuan_model_tokenizer(model_dir: str,
                                 load_model: bool = True,
                                 add_special_token: bool = True,
                                 torch_dtype: Dtype = torch.float16):
    """load from an independent repository"""
    model_config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code=True)
    model_config.torch_dtype = torch_dtype
    logger.info(f'model_config: {model_config}')
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True)
    model = None
    if load_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=model_config,
            device_map='auto',
            torch_dtype=torch_dtype,
            trust_remote_code=True)

    if add_special_token:
        _add_special_token(tokenizer)
    return model, tokenizer


def get_chatglm2_model_tokenizer(model_dir: str,
                                 load_model: bool = True,
                                 add_special_token: bool = True,
                                 torch_dtype: Dtype = torch.float16):
    """load from ms library"""
    config = read_config(model_dir)
    logger.info(config)
    model_config = ChatGLM2Config.from_pretrained(model_dir)
    model_config.torch_dtype = torch_dtype
    logger.info(model_config)
    tokenizer = ChatGLM2Tokenizer.from_pretrained(model_dir)
    model = None
    if load_model:
        model = Model.from_pretrained(
            model_dir,
            cfg_dict=config,
            config=model_config,
            device_map='auto',
            torch_dtype=torch_dtype)
    if add_special_token:
        _add_special_token(tokenizer)
    return model, tokenizer


def get_llama2_model_tokenizer(model_dir: str,
                               load_model: bool = True,
                               add_special_token: bool = True,
                               torch_dtype: Dtype = torch.float16):
    model_config = AutoConfig.from_pretrained(model_dir)
    model_config.torch_dtype = torch_dtype
    logger.info(model_config)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = None
    if load_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=model_config,
            device_map='auto',
            torch_dtype=torch_dtype)
    if add_special_token:
        _add_special_token(tokenizer)
    return model, tokenizer


# 'reference': 'https://modelscope.cn/models/{model_id}/summary'
class LoRATM(NamedTuple):
    # default lora target modules
    baichuan = ['W_pack']
    chatglm2 = ['query_key_value']
    llama2 = ['q_proj', 'k_proj', 'v_proj']


MODEL_MAPPER = {
    'baichuan-7b': {
        'model_id': 'baichuan-inc/baichuan-7B',
        'revision': 'v1.0.7',
        'get_function': get_baichuan_model_tokenizer,
        'lora_TM': LoRATM.baichuan
    },
    'baichuan-13b': {
        'model_id': 'baichuan-inc/Baichuan-13B-Base',
        'revision': 'v1.0.3',
        'get_function': get_baichuan_model_tokenizer,
        'lora_TM': LoRATM.baichuan
    },
    'chatglm2': {
        'model_id': 'ZhipuAI/chatglm2-6b',
        'revision': 'v1.0.6',
        'get_function': get_chatglm2_model_tokenizer,
        'lora_TM': LoRATM.chatglm2
    },
    'llama2-7b': {
        'model_id': 'modelscope/Llama-2-7b-ms',
        'revision': 'v1.0.2',
        'ignore_file_pattern': [r'.+\.bin$'],
        'get_function': get_llama2_model_tokenizer,
        'lora_TM': LoRATM.llama2
    },
    'llama2-13b': {
        'model_id': 'modelscope/Llama-2-13b-ms',
        'revision': 'v1.0.2',
        'ignore_file_pattern': [r'.+\.bin$'],
        'get_function': get_llama2_model_tokenizer,
        'lora_TM': LoRATM.llama2
    },
    'openbuddy-llama2-13b': {
        'model_id': 'OpenBuddy/openbuddy-llama2-13b-v8.1-fp16',
        'revision': 'master',
        'get_function': get_llama2_model_tokenizer,
        'lora_TM': LoRATM.llama2
    }
}


def get_model_tokenizer(model_type: str,
                        load_model: bool = True,
                        add_special_token: bool = True,
                        torch_dtype: Dtype = torch.float16):
    data = MODEL_MAPPER.get(model_type)
    if data is None:
        raise ValueError(f'model_type: {model_type}')
    model_id = data['model_id']
    revision = data['revision']
    get_function = data['get_function']
    ignore_file_pattern = data.get('ignore_file_pattern', [])
    model_dir = snapshot_download(
        model_id, revision, ignore_file_pattern=ignore_file_pattern)
    model, tokenizer = get_function(model_dir, load_model, add_special_token,
                                    torch_dtype)
    return model, tokenizer, model_dir
