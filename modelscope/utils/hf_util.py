# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import sys

from transformers import AutoConfig as AutoConfigHF
from transformers import AutoModel as AutoModelHF
from transformers import AutoModelForCausalLM as AutoModelForCausalLMHF
from transformers import AutoModelForSeq2SeqLM as AutoModelForSeq2SeqLMHF
from transformers import AutoTokenizer as AutoTokenizerHF
from transformers import GenerationConfig as GenerationConfigHF
from transformers import PreTrainedModel, PreTrainedTokenizerBase

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


def get_wrapped_class(module_class, ignore_file_pattern=[], **kwargs):
    """Get a custom wrapper class for  auto classes to download the models from the ModelScope hub
    Args:
        module_class: The actual module class
        ignore_file_pattern (`str` or `List`, *optional*, default to `None`):
            Any file pattern to be ignored in downloading, like exact file names or file extensions.
    Returns:
        The wrapper
    """

    class ClassWrapper(module_class):

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                            **kwargs):
            if not os.path.exists(pretrained_model_name_or_path):
                revision = kwargs.pop('revision', None)
                model_dir = snapshot_download(
                    pretrained_model_name_or_path,
                    revision=revision,
                    ignore_file_pattern=ignore_file_pattern,
                    user_agent=user_agent())
            else:
                model_dir = pretrained_model_name_or_path

            return module_class.from_pretrained(model_dir, *model_args,
                                                **kwargs)

    return ClassWrapper


AutoModel = get_wrapped_class(
    AutoModelHF, ignore_file_pattern=[r'\w+\.safetensors'])
AutoModelForCausalLM = get_wrapped_class(
    AutoModelForCausalLMHF, ignore_file_pattern=[r'\w+\.safetensors'])
AutoModelForSeq2SeqLM = get_wrapped_class(
    AutoModelForSeq2SeqLMHF, ignore_file_pattern=[r'\w+\.safetensors'])

AutoTokenizer = get_wrapped_class(
    AutoTokenizerHF, ignore_file_pattern=[r'\w+\.bin', r'\w+\.safetensors'])
AutoConfig = get_wrapped_class(
    AutoConfigHF, ignore_file_pattern=[r'\w+\.bin', r'\w+\.safetensors'])
GenerationConfig = get_wrapped_class(
    GenerationConfigHF, ignore_file_pattern=[r'\w+\.bin', r'\w+\.safetensors'])
