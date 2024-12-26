# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import tempfile
from functools import partial
from pathlib import Path
from types import MethodType
from typing import Dict, List, Optional, Union, BinaryIO
from urllib.error import HTTPError

from huggingface_hub.hf_api import CommitInfo, future_compatible
from transformers import AutoConfig as AutoConfigHF
from transformers import AutoFeatureExtractor as AutoFeatureExtractorHF
from transformers import AutoImageProcessor as AutoImageProcessorHF
from transformers import AutoModel as AutoModelHF
from transformers import \
    AutoModelForAudioClassification as AutoModelForAudioClassificationHF
from transformers import AutoModelForCausalLM as AutoModelForCausalLMHF
from transformers import \
    AutoModelForDocumentQuestionAnswering as \
    AutoModelForDocumentQuestionAnsweringHF
from transformers import \
    AutoModelForImageClassification as AutoModelForImageClassificationHF
from transformers import \
    AutoModelForImageSegmentation as AutoModelForImageSegmentationHF
from transformers import \
    AutoModelForInstanceSegmentation as AutoModelForInstanceSegmentationHF
from transformers import \
    AutoModelForMaskedImageModeling as AutoModelForMaskedImageModelingHF
from transformers import AutoModelForMaskedLM as AutoModelForMaskedLMHF
from transformers import \
    AutoModelForMaskGeneration as AutoModelForMaskGenerationHF
from transformers import \
    AutoModelForObjectDetection as AutoModelForObjectDetectionHF
from transformers import AutoModelForPreTraining as AutoModelForPreTrainingHF
from transformers import \
    AutoModelForQuestionAnswering as AutoModelForQuestionAnsweringHF
from transformers import \
    AutoModelForSemanticSegmentation as AutoModelForSemanticSegmentationHF
from transformers import AutoModelForSeq2SeqLM as AutoModelForSeq2SeqLMHF
from transformers import \
    AutoModelForSequenceClassification as AutoModelForSequenceClassificationHF
from transformers import \
    AutoModelForSpeechSeq2Seq as AutoModelForSpeechSeq2SeqHF
from transformers import \
    AutoModelForTableQuestionAnswering as AutoModelForTableQuestionAnsweringHF
from transformers import AutoModelForTextEncoding as AutoModelForTextEncodingHF
from transformers import \
    AutoModelForTokenClassification as AutoModelForTokenClassificationHF
from transformers import \
    AutoModelForUniversalSegmentation as AutoModelForUniversalSegmentationHF
from transformers import AutoModelForVision2Seq as AutoModelForVision2SeqHF
from transformers import \
    AutoModelForVisualQuestionAnswering as \
    AutoModelForVisualQuestionAnsweringHF
from transformers import \
    AutoModelForZeroShotImageClassification as \
    AutoModelForZeroShotImageClassificationHF
from transformers import \
    AutoModelForZeroShotObjectDetection as \
    AutoModelForZeroShotObjectDetectionHF
from transformers import AutoProcessor as AutoProcessorHF
from transformers import AutoTokenizer as AutoTokenizerHF
from transformers import BatchFeature as BatchFeatureHF
from transformers import BitsAndBytesConfig as BitsAndBytesConfigHF
from transformers import GenerationConfig as GenerationConfigHF
from transformers import (PretrainedConfig, PreTrainedModel,
                          PreTrainedTokenizerBase)
from transformers import T5EncoderModel as T5EncoderModelHF
from transformers import __version__ as transformers_version

from modelscope import push_to_hub, snapshot_download
from modelscope.utils.constant import DEFAULT_MODEL_REVISION, Invoke
from .logger import get_logger

try:
    from transformers import GPTQConfig as GPTQConfigHF
    from transformers import AwqConfig as AwqConfigHF
except ImportError:
    GPTQConfigHF = None
    AwqConfigHF = None

logger = get_logger()


class UnsupportedAutoClass:

    def __init__(self, name: str):
        self.error_msg =\
            f'{name} is not supported with your installed Transformers version {transformers_version}. ' + \
            'Please update your Transformers by "pip install transformers -U".'

    def from_pretrained(self, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        raise ImportError(self.error_msg)

    def from_config(self, cls, config):
        raise ImportError(self.error_msg)


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


def _whoami(self, token: Union[bool, str, None] = None) -> Dict:
    from modelscope.hub.api import ModelScopeConfig
    return {'name': ModelScopeConfig.get_user_info()[0] or 'unknown'}


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

    hf_api.whoami = MethodType(_whoami, api)
    huggingface_hub.whoami = hf_api.whoami
    huggingface_hub.hf_api.whoami = hf_api.whoami

    from huggingface_hub.repocard import RepoCard
    RepoCard.validate = lambda *args, **kwargs: None

    def create_repo(repo_id: str,
                    *,
                    token: Union[str, bool, None] = None,
                    private: bool = False,
                    **kwargs) -> 'RepoUrl':
        """
        Create a new repository on the hub.

        Args:
            repo_id: The ID of the repository to create.
            token: The authentication token to use.
            private: Whether the repository should be private.
            **kwargs: Additional arguments.

        Returns:
            RepoUrl: The URL of the created repository.
        """
        from modelscope.hub.create_model import create_model_repo
        hub_model_id = create_model_repo(repo_id, token, private)
        from huggingface_hub import RepoUrl
        return RepoUrl(url=hub_model_id, )

    @future_compatible
    def upload_folder(
        *,
        repo_id: str,
        folder_path: Union[str, Path],
        path_in_repo: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        token: Union[str, bool, None] = None,
        revision: Optional[str] = 'master',
        ignore_patterns: Optional[Union[List[str], str]] = None,
        **kwargs,
    ):
        from modelscope.hub.push_to_hub import push_model_to_hub
        push_model_to_hub(repo_id, folder_path, path_in_repo, commit_message,
                          commit_description, token, True, revision,
                          ignore_patterns)
        return CommitInfo(
            commit_url=f'https://www.modelscope.cn/models/{repo_id}/files',
            commit_message=commit_message,
            commit_description=commit_description,
            oid=None,
        )

    @future_compatible
    def upload_file(
        self,
        *,
        path_or_fileobj: Union[str, Path, bytes, BinaryIO],
        path_in_repo: str,
        repo_id: str,
        token: Union[str, bool, None] = None,
        revision: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        **kwargs,
    ):

        from modelscope.hub.push_to_hub import push_files_to_hub
        push_files_to_hub(path_or_fileobj, path_in_repo, repo_id, token, revision, commit_message, commit_description)


    huggingface_hub.create_repo = create_repo
    huggingface_hub.upload_folder = partial(upload_folder, api)

    hf_api.upload_file = MethodType(upload_file, api)
    huggingface_hub.upload_file = hf_api.upload_file
    huggingface_hub.hf_api.upload_file = hf_api.upload_file

    from transformers.utils import hub
    hub.create_repo = create_repo

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
            file_filter = None
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
AutoModelForVision2Seq = get_wrapped_class(AutoModelForVision2SeqHF)
AutoModelForSequenceClassification = get_wrapped_class(
    AutoModelForSequenceClassificationHF)
AutoModelForTokenClassification = get_wrapped_class(
    AutoModelForTokenClassificationHF)
AutoModelForImageSegmentation = get_wrapped_class(
    AutoModelForImageSegmentationHF)
AutoModelForImageClassification = get_wrapped_class(
    AutoModelForImageClassificationHF)
AutoModelForZeroShotImageClassification = get_wrapped_class(
    AutoModelForZeroShotImageClassificationHF)
try:
    from transformers import AutoModelForImageToImage as AutoModelForImageToImageHF
    AutoModelForImageToImage = get_wrapped_class(AutoModelForImageToImageHF)
except ImportError:
    AutoModelForImageToImage = UnsupportedAutoClass('AutoModelForImageToImage')

try:
    from transformers import AutoModelForImageTextToText as AutoModelForImageTextToTextHF
    AutoModelForImageTextToText = get_wrapped_class(
        AutoModelForImageTextToTextHF)
except ImportError:
    AutoModelForImageTextToText = UnsupportedAutoClass(
        'AutoModelForImageTextToText')

try:
    from transformers import AutoModelForKeypointDetection as AutoModelForKeypointDetectionHF
    AutoModelForKeypointDetection = get_wrapped_class(
        AutoModelForKeypointDetectionHF)
except ImportError:
    AutoModelForKeypointDetection = UnsupportedAutoClass(
        'AutoModelForKeypointDetection')

AutoModelForQuestionAnswering = get_wrapped_class(
    AutoModelForQuestionAnsweringHF)
AutoModelForTableQuestionAnswering = get_wrapped_class(
    AutoModelForTableQuestionAnsweringHF)
AutoModelForVisualQuestionAnswering = get_wrapped_class(
    AutoModelForVisualQuestionAnsweringHF)
AutoModelForDocumentQuestionAnswering = get_wrapped_class(
    AutoModelForDocumentQuestionAnsweringHF)
AutoModelForSemanticSegmentation = get_wrapped_class(
    AutoModelForSemanticSegmentationHF)
AutoModelForUniversalSegmentation = get_wrapped_class(
    AutoModelForUniversalSegmentationHF)
AutoModelForInstanceSegmentation = get_wrapped_class(
    AutoModelForInstanceSegmentationHF)
AutoModelForObjectDetection = get_wrapped_class(AutoModelForObjectDetectionHF)
AutoModelForZeroShotObjectDetection = get_wrapped_class(
    AutoModelForZeroShotObjectDetectionHF)
AutoModelForAudioClassification = get_wrapped_class(
    AutoModelForAudioClassificationHF)
AutoModelForSpeechSeq2Seq = get_wrapped_class(AutoModelForSpeechSeq2SeqHF)
AutoModelForMaskedImageModeling = get_wrapped_class(
    AutoModelForMaskedImageModelingHF)
AutoModelForMaskedLM = get_wrapped_class(AutoModelForMaskedLMHF)
AutoModelForMaskGeneration = get_wrapped_class(AutoModelForMaskGenerationHF)
AutoModelForPreTraining = get_wrapped_class(AutoModelForPreTrainingHF)
AutoModelForTextEncoding = get_wrapped_class(AutoModelForTextEncodingHF)
T5EncoderModel = get_wrapped_class(T5EncoderModelHF)
try:
    from transformers import \
        Qwen2VLForConditionalGeneration as Qwen2VLForConditionalGenerationHF
    Qwen2VLForConditionalGeneration = get_wrapped_class(
        Qwen2VLForConditionalGenerationHF)
except ImportError:
    Qwen2VLForConditionalGeneration = UnsupportedAutoClass(
        'Qwen2VLForConditionalGeneration')

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
