# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib
import os
from pathlib import Path
from types import MethodType
from typing import Dict, Literal, Optional, Union

from transformers import AutoConfig as AutoConfigHF
from transformers import AutoImageProcessor as AutoImageProcessorHF
from transformers import AutoModel as AutoModelHF
from transformers import AutoModelForCausalLM as AutoModelForCausalLMHF
from transformers import AutoModelForSeq2SeqLM as AutoModelForSeq2SeqLMHF
from transformers import \
    AutoModelForSequenceClassification as AutoModelForSequenceClassificationHF
from transformers import \
    AutoModelForTokenClassification as AutoModelForTokenClassificationHF
from transformers import AutoTokenizer as AutoTokenizerHF
from transformers import BatchFeature as BatchFeatureHF
from transformers import BitsAndBytesConfig as BitsAndBytesConfigHF
from transformers import GenerationConfig as GenerationConfigHF
from transformers import (PretrainedConfig, PreTrainedModel,
                          PreTrainedTokenizerBase)

from modelscope import snapshot_download
from modelscope.utils.constant import DEFAULT_MODEL_REVISION, Invoke

try:
    from transformers import GPTQConfig as GPTQConfigHF
    from transformers import AwqConfig as AwqConfigHF
except ImportError:
    GPTQConfigHF = None
    AwqConfigHF = None


def user_agent(invoked_by=None):
    if invoked_by is None:
        invoked_by = Invoke.PRETRAINED
    uagent = '%s/%s' % (Invoke.KEY, invoked_by)
    return uagent


def file_exists(
    self,
    repo_id: str,
    filename: str,
    *,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    token: Union[str, bool, None] = None,
):
    from modelscope.hub.api import HubApi
    api = HubApi()
    if token is None:
        token = os.environ.get('MODELSCOPE_API_TOKEN')
    if token:
        api.login(token)
    files = api.get_model_files(repo_id, revision=revision)
    files = [file['Name'] for file in files]
    return filename in files


def ms_hub_download(
    repo_id: str,
    filename: str,
    *,
    subfolder: Optional[str] = None,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    library_name: Optional[str] = None,
    library_version: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    local_dir: Union[str, Path, None] = None,
    user_agent: Union[Dict, str, None] = None,
    force_download: bool = False,
    proxies: Optional[Dict] = None,
    etag_timeout: float = 10,
    token: Union[bool, str, None] = None,
    local_files_only: bool = False,
    headers: Optional[Dict[str, str]] = None,
    endpoint: Optional[str] = None,
    # Deprecated args
    legacy_cache_layout: bool = False,
    resume_download: Optional[bool] = None,
    force_filename: Optional[str] = None,
    local_dir_use_symlinks: Union[bool, Literal['auto']] = 'auto',
):
    need_bin_files = '.safetensors' in filename or '.bin' in filename
    if not need_bin_files:
        ignore_file_pattern = [r'\w+\.bin', r'\w+\.safetensors']
    else:
        ignore_file_pattern = None

    if token is None:
        token = os.environ.get('MODELSCOPE_API_TOKEN')
    if token:
        from modelscope.hub.api import HubApi
        api = HubApi()
        api.login(token)

    model_dir = snapshot_download(
        repo_id,
        cache_dir=cache_dir,
        local_dir=local_dir,
        local_files_only=local_files_only,
        revision=revision,
        ignore_file_pattern=ignore_file_pattern)
    for dirs, _, files in os.walk(model_dir):
        if filename in files:
            return os.path.join(model_dir, dirs, filename)
    return None


def patch_pretrained_class():

    def get_model_dir(pretrained_model_name_or_path, ignore_file_pattern,
                      **kwargs):
        if not os.path.exists(pretrained_model_name_or_path):
            revision = kwargs.pop('revision', None)
            model_dir = snapshot_download(
                pretrained_model_name_or_path,
                revision=revision,
                ignore_file_pattern=ignore_file_pattern)
        else:
            model_dir = pretrained_model_name_or_path
        return model_dir

    def patch_tokenizer_base():
        """ Monkey patch PreTrainedTokenizerBase.from_pretrained to adapt to modelscope hub.
        """
        ori_from_pretrained = PreTrainedTokenizerBase.from_pretrained.__func__

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                            **kwargs):
            ignore_file_pattern = [r'\w+\.bin', r'\w+\.safetensors']
            model_dir = get_model_dir(pretrained_model_name_or_path,
                                      ignore_file_pattern, **kwargs)
            return ori_from_pretrained(cls, model_dir, *model_args, **kwargs)

        PreTrainedTokenizerBase.from_pretrained = from_pretrained

    def patch_config_base():
        """ Monkey patch PretrainedConfig.from_pretrained to adapt to modelscope hub.
        """
        ori_from_pretrained = PretrainedConfig.from_pretrained.__func__
        ori_get_config_dict = PretrainedConfig.get_config_dict.__func__

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                            **kwargs):
            ignore_file_pattern = [r'\w+\.bin', r'\w+\.safetensors']
            model_dir = get_model_dir(pretrained_model_name_or_path,
                                      ignore_file_pattern, **kwargs)
            return ori_from_pretrained(cls, model_dir, *model_args, **kwargs)

        @classmethod
        def get_config_dict(cls, pretrained_model_name_or_path, **kwargs):
            ignore_file_pattern = [r'\w+\.bin', r'\w+\.safetensors']
            model_dir = get_model_dir(pretrained_model_name_or_path,
                                      ignore_file_pattern, **kwargs)
            return ori_get_config_dict(cls, model_dir, **kwargs)

        PretrainedConfig.get_config_dict = get_config_dict

    def patch_model_base():
        """ Monkey patch PreTrainedModel.from_pretrained to adapt to modelscope hub.
        """
        ori_from_pretrained = PreTrainedModel.from_pretrained.__func__

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                            **kwargs):
            model_dir = get_model_dir(pretrained_model_name_or_path, None,
                                      **kwargs)
            return ori_from_pretrained(cls, model_dir, *model_args, **kwargs)

        PreTrainedModel.from_pretrained = from_pretrained

    patch_tokenizer_base()
    patch_config_base()
    patch_model_base()


def patch_hub():
    import huggingface_hub
    from huggingface_hub import hf_api
    from huggingface_hub.hf_api import api

    huggingface_hub.hf_hub_download = ms_hub_download
    huggingface_hub.file_download.hf_hub_download = ms_hub_download

    hf_api.file_exists = MethodType(file_exists, api)
    huggingface_hub.file_exists = hf_api.file_exists
    huggingface_hub.hf_api.file_exists = hf_api.file_exists

    patch_pretrained_class()

    if importlib.util.find_spec('vllm') is not None:
        try:
            from vllm.transformers_utils import config
            config.file_exists = hf_api.file_exists
            config.hf_hub_download = ms_hub_download
        except (ImportError, AttributeError, ModuleNotFoundError):
            pass


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
                revision = kwargs.pop('revision', DEFAULT_MODEL_REVISION)
                model_dir = snapshot_download(
                    pretrained_model_name_or_path,
                    revision=revision,
                    ignore_file_pattern=ignore_file_pattern,
                    user_agent=user_agent())
            else:
                model_dir = pretrained_model_name_or_path

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
AwqConfig = AwqConfigHF
BitsAndBytesConfig = BitsAndBytesConfigHF
AutoImageProcessor = get_wrapped_class(AutoImageProcessorHF)
BatchFeature = get_wrapped_class(BatchFeatureHF)
