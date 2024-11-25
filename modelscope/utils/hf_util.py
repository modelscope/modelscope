# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from pathlib import Path
from types import MethodType
from typing import Optional, Union

from transformers import AutoConfig as AutoConfigHF
from transformers import AutoFeatureExtractor as AutoFeatureExtractorHF
from transformers import AutoImageProcessor as AutoImageProcessorHF
from transformers import AutoModel as AutoModelHF
from transformers import AutoModelForCausalLM as AutoModelForCausalLMHF
from transformers import \
    AutoModelForImageClassification as AutoModelForImageClassificationHF
from transformers import \
    AutoModelForImageSegmentation as AutoModelForImageSegmentationHF
from transformers import \
    AutoModelForImageTextToText as AutoModelForImageTextToTextHF
from transformers import AutoModelForImageToImage as AutoModelForImageToImageHF
from transformers import AutoModelForMaskedLM as AutoModelForMaskedLMHF
from transformers import \
    AutoModelForMaskGeneration as AutoModelForMaskGenerationHF
from transformers import AutoModelForPreTraining as AutoModelForPreTrainingHF
from transformers import \
    AutoModelForQuestionAnswering as AutoModelForQuestionAnsweringHF
from transformers import AutoModelForSeq2SeqLM as AutoModelForSeq2SeqLMHF
from transformers import \
    AutoModelForSequenceClassification as AutoModelForSequenceClassificationHF
from transformers import AutoModelForTextEncoding as AutoModelForTextEncodingHF
from transformers import \
    AutoModelForTokenClassification as AutoModelForTokenClassificationHF
from transformers import AutoProcessor as AutoProcessorHF
from transformers import AutoTokenizer as AutoTokenizerHF
from transformers import BatchFeature as BatchFeatureHF
from transformers import BitsAndBytesConfig as BitsAndBytesConfigHF
from transformers import GenerationConfig as GenerationConfigHF
from transformers import (PretrainedConfig, PreTrainedModel,
                          PreTrainedTokenizerBase)
from transformers import T5EncoderModel as T5EncoderModelHF

from modelscope import snapshot_download
from modelscope.utils.constant import DEFAULT_MODEL_REVISION, Invoke
from .logger import get_logger

try:
    from transformers import GPTQConfig as GPTQConfigHF
    from transformers import AwqConfig as AwqConfigHF
except ImportError:
    GPTQConfigHF = None
    AwqConfigHF = None

logger = get_logger()


def user_agent(invoked_by=None):
    if invoked_by is None:
        invoked_by = Invoke.PRETRAINED
    uagent = '%s/%s' % (Invoke.KEY, invoked_by)
    return uagent


def _try_login(token: Optional[str] = None):
    from modelscope.hub.api import HubApi
    api = HubApi()
    if token is None:
        token = os.environ.get('MODELSCOPE_API_TOKEN')
    if token:
        api.login(token)


def _file_exists(
    self,
    repo_id: str,
    filename: str,
    *,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    token: Union[str, bool, None] = None,
):
    """Patch huggingface_hub.file_exists"""
    if repo_type is not None:
        logger.warning(
            'The passed in repo_type will not be used in modelscope. Now only model repo can be queried.'
        )
    _try_login(token)
    from modelscope.hub.api import HubApi
    api = HubApi()
    return api.file_exists(repo_id, filename, revision=revision)


def _file_download(repo_id: str,
                   filename: str,
                   *,
                   subfolder: Optional[str] = None,
                   repo_type: Optional[str] = None,
                   revision: Optional[str] = None,
                   cache_dir: Union[str, Path, None] = None,
                   local_dir: Union[str, Path, None] = None,
                   token: Union[bool, str, None] = None,
                   local_files_only: bool = False,
                   **kwargs):
    """Patch huggingface_hub.hf_hub_download"""
    if len(kwargs) > 0:
        logger.warning(
            'The passed in library_name,library_version,user_agent,force_download,proxies'
            'etag_timeout,headers,endpoint '
            'will not be used in modelscope.')
    assert repo_type in (
        None, 'model',
        'dataset'), f'repo_type={repo_type} is not supported in ModelScope'
    if repo_type in (None, 'model'):
        from modelscope.hub.file_download import model_file_download as file_download
    else:
        from modelscope.hub.file_download import dataset_file_download as file_download
    _try_login(token)
    return file_download(
        repo_id,
        file_path=os.path.join(subfolder, filename) if subfolder else filename,
        cache_dir=cache_dir,
        local_dir=local_dir,
        local_files_only=local_files_only,
        revision=revision)


def _patch_pretrained_class():

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
            ignore_file_pattern = [
                r'\w+\.bin', r'\w+\.safetensors', r'\w+\.pth', r'\w+\.pt'
            ]
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
            ignore_file_pattern = [
                r'\w+\.bin', r'\w+\.safetensors', r'\w+\.pth', r'\w+\.pt'
            ]
            model_dir = get_model_dir(pretrained_model_name_or_path,
                                      ignore_file_pattern, **kwargs)
            return ori_from_pretrained(cls, model_dir, *model_args, **kwargs)

        @classmethod
        def get_config_dict(cls, pretrained_model_name_or_path, **kwargs):
            ignore_file_pattern = [
                r'\w+\.bin', r'\w+\.safetensors', r'\w+\.pth', r'\w+\.pt'
            ]
            model_dir = get_model_dir(pretrained_model_name_or_path,
                                      ignore_file_pattern, **kwargs)
            return ori_get_config_dict(cls, model_dir, **kwargs)

        PretrainedConfig.from_pretrained = from_pretrained
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

    def patch_image_processor_base():
        """ Monkey patch AutoImageProcessorHF.from_pretrained to adapt to modelscope hub.
        """
        ori_from_pretrained = AutoImageProcessorHF.from_pretrained.__func__

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                            **kwargs):
            model_dir = get_model_dir(pretrained_model_name_or_path, None,
                                      **kwargs)
            return ori_from_pretrained(cls, model_dir, *model_args, **kwargs)

        AutoImageProcessorHF.from_pretrained = from_pretrained

    def patch_auto_processor_base():
        """ Monkey patch AutoProcessorHF.from_pretrained to adapt to modelscope hub.
        """
        ori_from_pretrained = AutoProcessorHF.from_pretrained.__func__

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                            **kwargs):
            model_dir = get_model_dir(pretrained_model_name_or_path, None,
                                      **kwargs)
            return ori_from_pretrained(cls, model_dir, *model_args, **kwargs)

        AutoProcessorHF.from_pretrained = from_pretrained

    def patch_feature_extractor_base():
        """ Monkey patch AutoFeatureExtractorHF.from_pretrained to adapt to modelscope hub.
        """
        ori_from_pretrained = AutoFeatureExtractorHF.from_pretrained.__func__

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                            **kwargs):
            model_dir = get_model_dir(pretrained_model_name_or_path, None,
                                      **kwargs)
            return ori_from_pretrained(cls, model_dir, *model_args, **kwargs)

        AutoFeatureExtractorHF.from_pretrained = from_pretrained

    patch_tokenizer_base()
    patch_config_base()
    patch_model_base()
    patch_image_processor_base()
    patch_auto_processor_base()
    patch_feature_extractor_base()


def patch_hub():
    """Patch hf hub, which to make users can download models from modelscope to speed up.
    """
    import huggingface_hub
    from huggingface_hub import hf_api
    from huggingface_hub.hf_api import api

    huggingface_hub.hf_hub_download = _file_download
    huggingface_hub.file_download.hf_hub_download = _file_download

    hf_api.file_exists = MethodType(_file_exists, api)
    huggingface_hub.file_exists = hf_api.file_exists
    huggingface_hub.hf_api.file_exists = hf_api.file_exists

    _patch_pretrained_class()


def get_wrapped_class(module_class,
                      ignore_file_pattern=[],
                      file_filter=None,
                      **kwargs):
    """Get a custom wrapper class for  auto classes to download the models from the ModelScope hub
    Args:
        module_class: The actual module class
        ignore_file_pattern (`str` or `List`, *optional*, default to `None`):
            Any file pattern to be ignored in downloading, like exact file names or file extensions.
    Returns:
        The wrapper
    """
    default_ignore_file_pattern = ignore_file_pattern
    default_file_filter = file_filter

    class ClassWrapper(module_class):

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                            **kwargs):
            ignore_file_pattern = kwargs.pop('ignore_file_pattern',
                                             default_ignore_file_pattern)
            subfolder = kwargs.pop('subfolder', default_file_filter)

            if subfolder:
                file_filter = f'{subfolder}/*'
            if not os.path.exists(pretrained_model_name_or_path):
                revision = kwargs.pop('revision', DEFAULT_MODEL_REVISION)
                if file_filter is None:
                    model_dir = snapshot_download(
                        pretrained_model_name_or_path,
                        revision=revision,
                        ignore_file_pattern=ignore_file_pattern,
                        user_agent=user_agent())
                else:
                    model_dir = os.path.join(
                        snapshot_download(
                            pretrained_model_name_or_path,
                            revision=revision,
                            ignore_file_pattern=ignore_file_pattern,
                            allow_file_pattern=file_filter,
                            user_agent=user_agent()), subfolder)
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
AutoModelForImageSegmentation = get_wrapped_class(
    AutoModelForImageSegmentationHF)
AutoModelForImageClassification = get_wrapped_class(
    AutoModelForImageClassificationHF)
AutoModelForImageTextToText = get_wrapped_class(AutoModelForImageTextToTextHF)
AutoModelForImageToImage = get_wrapped_class(AutoModelForImageToImageHF)
AutoModelForQuestionAnswering = get_wrapped_class(
    AutoModelForQuestionAnsweringHF)
AutoModelForMaskedLM = get_wrapped_class(AutoModelForMaskedLMHF)
AutoModelForMaskGeneration = get_wrapped_class(AutoModelForMaskGenerationHF)
AutoModelForPreTraining = get_wrapped_class(AutoModelForPreTrainingHF)
AutoModelForTextEncoding = get_wrapped_class(AutoModelForTextEncodingHF)
T5EncoderModel = get_wrapped_class(T5EncoderModelHF)

AutoTokenizer = get_wrapped_class(
    AutoTokenizerHF,
    ignore_file_pattern=[
        r'\w+\.bin', r'\w+\.safetensors', r'\w+\.pth', r'\w+\.pt', r'\w+\.h5'
    ])
AutoProcessor = get_wrapped_class(
    AutoProcessorHF,
    ignore_file_pattern=[
        r'\w+\.bin', r'\w+\.safetensors', r'\w+\.pth', r'\w+\.pt', r'\w+\.h5'
    ])
AutoConfig = get_wrapped_class(
    AutoConfigHF,
    ignore_file_pattern=[
        r'\w+\.bin', r'\w+\.safetensors', r'\w+\.pth', r'\w+\.pt', r'\w+\.h5'
    ])
GenerationConfig = get_wrapped_class(
    GenerationConfigHF,
    ignore_file_pattern=[
        r'\w+\.bin', r'\w+\.safetensors', r'\w+\.pth', r'\w+\.pt', r'\w+\.h5'
    ])
BitsAndBytesConfig = get_wrapped_class(
    BitsAndBytesConfigHF,
    ignore_file_pattern=[
        r'\w+\.bin', r'\w+\.safetensors', r'\w+\.pth', r'\w+\.pt', r'\w+\.h5'
    ])
AutoImageProcessor = get_wrapped_class(
    AutoImageProcessorHF,
    ignore_file_pattern=[
        r'\w+\.bin', r'\w+\.safetensors', r'\w+\.pth', r'\w+\.pt', r'\w+\.h5'
    ])

GPTQConfig = GPTQConfigHF
AwqConfig = AwqConfigHF
BatchFeature = get_wrapped_class(BatchFeatureHF)
