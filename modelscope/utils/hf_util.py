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
from transformers import GenerationConfig as GenerationConfigHF
from transformers import (PretrainedConfig, PreTrainedModel,
                          PreTrainedTokenizerBase)
from transformers.models.auto.tokenization_auto import (
    TOKENIZER_MAPPING_NAMES, get_tokenizer_config)

from modelscope import snapshot_download
from modelscope.utils.constant import Invoke


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


def check_hf_code(model_dir: str, auto_class: type) -> None:
    config_path = os.path.join(model_dir, 'config.json')
    auto_class_name = auto_class.__name__
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'{config_path} is not found')

    config_dict = PretrainedConfig.get_config_dict(config_path)[0]
    model_type = config_dict['model_type']
    if auto_class is AutoConfigHF:
        if model_type in CONFIG_MAPPING:
            return
    elif auto_class is AutoTokenizerHF:
        if model_type in TOKENIZER_MAPPING_NAMES:
            return
    else:
        mapping_names = [
            m.model_type for m in auto_class._model_mapping.keys()
        ]
        if model_type in mapping_names:
            return

    if auto_class is AutoTokenizerHF:
        tokenizer_config_dict = get_tokenizer_config(model_dir)
        auto_map = tokenizer_config_dict.get('auto_map', None)
        if auto_map is None:
            raise ValueError(f'`auto_map` key is not exists in {config_path}')
        module_name = auto_map.get(auto_class_name, None)[0]
    else:
        auto_map = config_dict.get('auto_map', None)
        if auto_map is None:
            raise ValueError(f'`auto_map` key is not exists in {config_path}')
        module_name = auto_map.get(auto_class_name, None)
    if module_name is None:
        raise ValueError(
            f'`{auto_class_name}` is not exists in `auto_map` of {config_path}'
        )
    module_path = os.path.join(model_dir, module_name.split('.')[0] + '.py')
    if not os.path.exists(module_path):
        raise FileNotFoundError(f'{module_path} is not found')


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
                check_hf_code(model_dir, module_class)
            module_obj = module_class.from_pretrained(model_dir, *model_args,
                                                      **kwargs)
            if module_class.__name__.startswith('AutoModel'):
                module_obj.model_dir = model_dir
            return module_obj

    ClassWrapper.__name__ = module_class.__name__
    ClassWrapper.__qualname__ = module_class.__qualname__
    return ClassWrapper


AutoModel = get_wrapped_class(
    AutoModelHF, ignore_file_pattern=[r'\w+\.safetensors'])
AutoModelForCausalLM = get_wrapped_class(
    AutoModelForCausalLMHF, ignore_file_pattern=[r'\w+\.safetensors'])
AutoModelForSeq2SeqLM = get_wrapped_class(
    AutoModelForSeq2SeqLMHF, ignore_file_pattern=[r'\w+\.safetensors'])
AutoModelForSequenceClassification = get_wrapped_class(
    AutoModelForSequenceClassificationHF,
    ignore_file_pattern=[r'\w+\.safetensors'])
AutoModelForTokenClassification = get_wrapped_class(
    AutoModelForTokenClassificationHF,
    ignore_file_pattern=[r'\w+\.safetensors'])

AutoTokenizer = get_wrapped_class(
    AutoTokenizerHF, ignore_file_pattern=[r'\w+\.bin', r'\w+\.safetensors'])
AutoConfig = get_wrapped_class(
    AutoConfigHF, ignore_file_pattern=[r'\w+\.bin', r'\w+\.safetensors'])
GenerationConfig = get_wrapped_class(
    GenerationConfigHF, ignore_file_pattern=[r'\w+\.bin', r'\w+\.safetensors'])
