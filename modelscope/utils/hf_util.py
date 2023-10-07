# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import sys

from transformers import CONFIG_MAPPING
from transformers import AutoConfig as AutoConfigHF
from transformers import AutoModel as AutoModelHF
from transformers import AutoModelForCausalLM as AutoModelForCausalLMHF
from transformers import AutoModelForSeq2SeqLM as AutoModelForSeq2SeqLMHF
from transformers import \
    AutoModelForSequenceClassification as AutoModelForSequenceClassificationHF
from transformers import \
    AutoModelForTokenClassification as AutoModelForTokenClassificationHF
from transformers import AutoTokenizer as AutoTokenizerHF
from transformers import BitsAndBytesConfig as BitsAndBytesConfigHF
from transformers import GenerationConfig as GenerationConfigHF
from transformers import (PretrainedConfig, PreTrainedModel,
                          PreTrainedTokenizerBase)
from transformers.models.auto.tokenization_auto import (
    TOKENIZER_MAPPING_NAMES, get_tokenizer_config)

from modelscope import snapshot_download
from modelscope.utils.constant import Invoke

try:
    from transformers import GPTQConfig as GPTQConfigHF
except ImportError:
    GPTQConfigHF = None


def user_agent(invoked_by=None):
    if invoked_by is None:
        invoked_by = Invoke.PRETRAINED
    uagent = '%s/%s' % (Invoke.KEY, invoked_by)
    return uagent


def patch_tokenizer_base():
    """ Monkey patch PreTrainedTokenizerBase.from_pretrained to adapt to modelscope hub.
    """
    ori_from_pretrained = PreTrainedTokenizerBase.from_pretrained.__func__

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        ignore_file_pattern = [r'\w+\.bin', r'\w+\.safetensors']
        if not os.path.exists(pretrained_model_name_or_path):
            revision = kwargs.pop('revision', None)
            model_dir = snapshot_download(
                pretrained_model_name_or_path,
                revision=revision,
                ignore_file_pattern=ignore_file_pattern)
        else:
            model_dir = pretrained_model_name_or_path
        return ori_from_pretrained(cls, model_dir, *model_args, **kwargs)

    PreTrainedTokenizerBase.from_pretrained = from_pretrained


def patch_model_base():
    """ Monkey patch PreTrainedModel.from_pretrained to adapt to modelscope hub.
    """
    ori_from_pretrained = PreTrainedModel.from_pretrained.__func__

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        ignore_file_pattern = [r'\w+\.safetensors']
        if not os.path.exists(pretrained_model_name_or_path):
            revision = kwargs.pop('revision', None)
            model_dir = snapshot_download(
                pretrained_model_name_or_path,
                revision=revision,
                ignore_file_pattern=ignore_file_pattern)
        else:
            model_dir = pretrained_model_name_or_path
        return ori_from_pretrained(cls, model_dir, *model_args, **kwargs)

    PreTrainedModel.from_pretrained = from_pretrained


patch_tokenizer_base()
patch_model_base()


def check_hf_code(model_dir: str, auto_class: type,
                  trust_remote_code: bool) -> None:
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'{config_path} is not found')
    config_dict = PretrainedConfig.get_config_dict(config_path)[0]
    auto_class_name = auto_class.__name__
    if auto_class is AutoTokenizerHF:
        tokenizer_config = get_tokenizer_config(model_dir)
    # load from repo
    if trust_remote_code:
        has_remote_code = False
        if auto_class is AutoTokenizerHF:
            auto_map = tokenizer_config.get('auto_map', None)
            if auto_map is not None:
                module_name = auto_map.get(auto_class_name, None)
                if module_name is not None:
                    module_name = module_name[0]
                    has_remote_code = True
        else:
            auto_map = config_dict.get('auto_map', None)
            if auto_map is not None:
                module_name = auto_map.get(auto_class_name, None)
                has_remote_code = module_name is not None

        if has_remote_code:
            module_path = os.path.join(model_dir,
                                       module_name.split('.')[0] + '.py')
            if not os.path.exists(module_path):
                raise FileNotFoundError(f'{module_path} is not found')
            return

    # trust_remote_code is False or has_remote_code is False
    model_type = config_dict.get('model_type', None)
    if model_type is None:
        raise ValueError(f'`model_type` key is not found in {config_path}.')

    trust_remote_code_info = '.'
    if not trust_remote_code:
        trust_remote_code_info = ', You can try passing `trust_remote_code=True`.'
    if auto_class is AutoConfigHF:
        if model_type not in CONFIG_MAPPING:
            raise ValueError(
                f'{model_type} not found in HF `CONFIG_MAPPING`{trust_remote_code_info}'
            )
    elif auto_class is AutoTokenizerHF:
        tokenizer_class = tokenizer_config.get('tokenizer_class')
        if tokenizer_class is not None:
            return
        if model_type not in TOKENIZER_MAPPING_NAMES:
            raise ValueError(
                f'{model_type} not found in HF `TOKENIZER_MAPPING_NAMES`{trust_remote_code_info}'
            )
    else:
        mapping_names = [
            m.model_type for m in auto_class._model_mapping.keys()
        ]
        if model_type not in mapping_names:
            raise ValueError(
                f'{model_type} not found in HF `auto_class._model_mapping`{trust_remote_code_info}'
            )


def get_wrapped_class(module_class, ignore_file_pattern=[], **kwargs):
    """Get a custom wrapper class for  auto classes to download the models from the ModelScope hub
    Args:
        module_class: The actual module class
        ignore_file_pattern (`str` or `List`, *optional*, default to `None`):
            Any file pattern to be ignored in downloading, like exact file names or file extensions.
    Returns:
        The wrapper
    """
    default_ignore_file_pattern = ignore_file_pattern

    class ClassWrapper(module_class):

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                            **kwargs):
            ignore_file_pattern = kwargs.pop('ignore_file_pattern',
                                             default_ignore_file_pattern)
            if not os.path.exists(pretrained_model_name_or_path):
                revision = kwargs.pop('revision', None)
                model_dir = snapshot_download(
                    pretrained_model_name_or_path,
                    revision=revision,
                    ignore_file_pattern=ignore_file_pattern,
                    user_agent=user_agent())
            else:
                model_dir = pretrained_model_name_or_path

            if module_class is not GenerationConfigHF:
                trust_remote_code = kwargs.get('trust_remote_code', False)
                check_hf_code(model_dir, module_class, trust_remote_code)
            module_obj = module_class.from_pretrained(model_dir, *model_args,
                                                      **kwargs)

            if module_class.__name__.startswith('AutoModel'):
                module_obj.model_dir = model_dir
            return module_obj

    ClassWrapper.__name__ = module_class.__name__
    ClassWrapper.__qualname__ = module_class.__qualname__
    return ClassWrapper


AutoModel = get_wrapped_class(AutoModelHF)
AutoModelForCausalLM = get_wrapped_class(AutoModelForCausalLMHF)
AutoModelForSeq2SeqLM = get_wrapped_class(AutoModelForSeq2SeqLMHF)
AutoModelForSequenceClassification = get_wrapped_class(
    AutoModelForSequenceClassificationHF)
AutoModelForTokenClassification = get_wrapped_class(
    AutoModelForTokenClassificationHF)

AutoTokenizer = get_wrapped_class(
    AutoTokenizerHF, ignore_file_pattern=[r'\w+\.bin', r'\w+\.safetensors'])
AutoConfig = get_wrapped_class(
    AutoConfigHF, ignore_file_pattern=[r'\w+\.bin', r'\w+\.safetensors'])
GenerationConfig = get_wrapped_class(
    GenerationConfigHF, ignore_file_pattern=[r'\w+\.bin', r'\w+\.safetensors'])
GPTQConfig = GPTQConfigHF
BitsAndBytesConfig = BitsAndBytesConfigHF
