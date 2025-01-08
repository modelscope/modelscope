# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib
import inspect
import os
import sys
from functools import partial
from pathlib import Path
from types import MethodType
from typing import BinaryIO, Dict, List, Optional, Union

from huggingface_hub.hf_api import CommitInfo, future_compatible

from modelscope import snapshot_download
from modelscope.utils.logger import get_logger

logger = get_logger()


extra_modules = ['T5']
lazy_module = sys.modules['transformers']
all_modules = lazy_module._modules
all_imported_modules = []
for module in all_modules:
    if 'auto' in module.lower() or any(m in module for m in extra_modules):
        all_imported_modules.append(importlib.import_module(f'transformers.{module}'))


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

    ignore_file_pattern = [
        r'\w+\.bin', r'\w+\.safetensors', r'\w+\.pth', r'\w+\.pt'
    ]

    def patch_pretrained_model_name_or_path(cls, pretrained_model_name_or_path,
                                            *model_args, **kwargs):
        model_dir = get_model_dir(pretrained_model_name_or_path,
                                  kwargs.pop('ignore_file_pattern', None),
                                  **kwargs)
        return kwargs.pop('ori_func')(cls, model_dir, *model_args, **kwargs)

    def patch_peft_model_id(cls, model, model_id, *model_args, **kwargs):
        model_dir = get_model_dir(model_id,
                                  kwargs.pop('ignore_file_pattern', None),
                                  **kwargs)
        return kwargs.pop('ori_func')(cls, model, model_dir, *model_args,
                                      **kwargs)

    def _get_peft_type(cls, model_id, **kwargs):
        model_dir = get_model_dir(model_id, ignore_file_pattern, **kwargs)
        return kwargs.pop('ori_func')(cls, model_dir, **kwargs)

    for var in all_imported_modules:
        if var is None:
            continue
        name = var.__name__
        need_model = 'model' in name.lower() or 'processor' in name.lower() or 'extractor' in name.lower()
        if need_model:
            ignore_file_pattern_kwargs = {}
        else:
            ignore_file_pattern_kwargs = {'ignore_file_pattern': ignore_file_pattern}

        has_from_pretrained = hasattr(var, 'from_pretrained')
        has_get_peft_type = hasattr(var, '_get_peft_type')
        has_get_config_dict = hasattr(var, 'get_config_dict')
        parameters = inspect.signature(var.from_pretrained).parameters
        is_peft = 'model' in parameters and 'model_id' in parameters
        if has_from_pretrained and not hasattr(var, '_from_pretrained_origin'):
            var._from_pretrained_origin = var.from_pretrained
            if not is_peft:
                var.from_pretrained = partial(patch_pretrained_model_name_or_path,
                                              ori_func=var._from_pretrained_origin,
                                              **ignore_file_pattern_kwargs)
            else:
                var.from_pretrained = partial(patch_peft_model_id,
                                              ori_func=var._from_pretrained_origin,
                                              **ignore_file_pattern_kwargs)
            delattr(var, '_from_pretrained_origin')
        if has_get_peft_type and not hasattr(var, '_get_peft_type_origin'):
            var._get_peft_type_origin = var._get_peft_type
            var._get_peft_type = partial(_get_peft_type,
                                          ori_func=var._get_peft_type_origin,
                                          **ignore_file_pattern_kwargs)
            delattr(var, '_get_peft_type_origin')

        if has_get_config_dict and not hasattr(var, '_get_config_dict_origin'):
            var._get_config_dict_origin = var.get_config_dict
            var.get_config_dict = partial(patch_pretrained_model_name_or_path,
                                          ori_func=var._get_config_dict_origin,
                                          **ignore_file_pattern_kwargs)
            delattr(var, '_get_config_dict_origin')


def _unpatch_pretrained_class():
    for var in all_imported_modules:
        if var is None:
            continue

        has_from_pretrained = hasattr(var, 'from_pretrained')
        has_get_peft_type = hasattr(var, '_get_peft_type')
        has_get_config_dict = hasattr(var, 'get_config_dict')
        if has_from_pretrained and hasattr(var, '_from_pretrained_origin'):
            var.from_pretrained = var._from_pretrained_origin
        if has_get_peft_type and hasattr(var, '_get_peft_type_origin'):
            var._get_peft_type = var._get_peft_type_origin
        if has_get_config_dict and hasattr(var, '_get_config_dict_origin'):
            var.get_config_dict = var._get_config_dict_origin


def _patch_hub():
    import huggingface_hub
    from huggingface_hub import hf_api
    from huggingface_hub.hf_api import api

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
        from modelscope.hub.api import HubApi
        api = HubApi()
        api.try_login(token)
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
        from modelscope import HubApi
        api = HubApi()
        api.try_login(token)
        return file_download(
            repo_id,
            file_path=os.path.join(subfolder, filename) if subfolder else filename,
            cache_dir=cache_dir,
            local_dir=local_dir,
            local_files_only=local_files_only,
            revision=revision)

    def _whoami(self, token: Union[bool, str, None] = None) -> Dict:
        from modelscope.hub.api import ModelScopeConfig
        from modelscope.hub.api import HubApi
        api = HubApi()
        api.try_login(token)
        return {'name': ModelScopeConfig.get_user_info()[0] or 'unknown'}

    def create_repo(self,
                    repo_id: str,
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
        push_files_to_hub(path_or_fileobj, path_in_repo, repo_id, token,
                          revision, commit_message, commit_description)

    # Patch repocard.validate
    from huggingface_hub import repocard
    if not hasattr(repocard.RepoCard, '_validate_origin'):
        repocard.RepoCard._validate_origin = repocard.RepoCard.validate
        repocard.RepoCard.validate = lambda *args, **kwargs: None

    if not hasattr(hf_api, '_hf_hub_download_origin'):
        # Patch hf_hub_download
        hf_api._hf_hub_download_origin = huggingface_hub.file_download.hf_hub_download
        huggingface_hub.hf_hub_download = _file_download
        huggingface_hub.file_download.hf_hub_download = _file_download

    if not hasattr(hf_api, '_file_exists_origin'):
        # Patch file_exists
        hf_api._file_exists_origin = hf_api.file_exists
        hf_api.file_exists = MethodType(_file_exists, api)
        huggingface_hub.file_exists = hf_api.file_exists
        huggingface_hub.hf_api.file_exists = hf_api.file_exists

    if not hasattr(hf_api, '_whoami_origin'):
        # Patch whoami
        hf_api._whoami_origin = hf_api.whoami
        hf_api.whoami = MethodType(_whoami, api)
        huggingface_hub.whoami = hf_api.whoami
        huggingface_hub.hf_api.whoami = hf_api.whoami

    if not hasattr(hf_api, '_create_repo_origin'):
        # Patch create_repo
        from transformers.utils import hub
        hf_api._create_repo_origin = hf_api.create_repo
        hf_api.create_repo = MethodType(create_repo, api)
        huggingface_hub.create_repo = hf_api.create_repo
        huggingface_hub.hf_api.create_repo = hf_api.create_repo
        hub.create_repo = hf_api.create_repo

    if not hasattr(hf_api, '_upload_folder_origin'):
        # Patch upload_folder
        hf_api._upload_folder_origin = hf_api.upload_folder
        hf_api.upload_folder = MethodType(upload_folder, api)
        huggingface_hub.upload_folder = hf_api.upload_folder
        huggingface_hub.hf_api.upload_folder = hf_api.upload_folder

    if not hasattr(hf_api, '_upload_file_origin'):
        # Patch upload_file
        hf_api._upload_file_origin = hf_api.upload_file
        hf_api.upload_file = MethodType(upload_file, api)
        huggingface_hub.upload_file = hf_api.upload_file
        huggingface_hub.hf_api.upload_file = hf_api.upload_file
        repocard.upload_file = hf_api.upload_file


def _unpatch_hub():
    import huggingface_hub
    from huggingface_hub import hf_api

    from huggingface_hub import repocard
    if hasattr(repocard.RepoCard, '_validate_origin'):
        repocard.RepoCard.validate = repocard.RepoCard._validate_origin
        delattr(repocard.RepoCard, '_validate_origin')

    if hasattr(hf_api, '_hf_hub_download_origin'):
        huggingface_hub.file_download.hf_hub_download = hf_api._hf_hub_download_origin
        huggingface_hub.hf_hub_download = hf_api._hf_hub_download_origin
        huggingface_hub.file_download.hf_hub_download = hf_api._hf_hub_download_origin
        delattr(hf_api, '_hf_hub_download_origin')

    if hasattr(hf_api, '_file_exists_origin'):
        hf_api.file_exists = hf_api._file_exists_origin
        huggingface_hub.file_exists = hf_api.file_exists
        huggingface_hub.hf_api.file_exists = hf_api.file_exists
        delattr(hf_api, '_file_exists_origin')

    if hasattr(hf_api, '_whoami_origin'):
        hf_api.whoami = hf_api._whoami_origin
        huggingface_hub.whoami = hf_api.whoami
        huggingface_hub.hf_api.whoami = hf_api.whoami
        delattr(hf_api, '_whoami_origin')

    if hasattr(hf_api, '_create_repo_origin'):
        from transformers.utils import hub
        hf_api.create_repo = hf_api._create_repo_origin
        huggingface_hub.create_repo = hf_api.create_repo
        huggingface_hub.hf_api.create_repo = hf_api.create_repo
        hub.create_repo = hf_api.create_repo
        delattr(hf_api, '_create_repo_origin')

    if hasattr(hf_api, '_upload_folder_origin'):
        hf_api.upload_folder = hf_api._upload_folder_origin
        huggingface_hub.upload_folder = hf_api.upload_folder
        huggingface_hub.hf_api.upload_folder = hf_api.upload_folder
        delattr(hf_api, '_upload_folder_origin')

    if hasattr(hf_api, '_upload_file_origin'):
        hf_api.upload_file = hf_api._upload_file_origin
        huggingface_hub.upload_file = hf_api.upload_file
        huggingface_hub.hf_api.upload_file = hf_api.upload_file
        repocard.upload_file = hf_api.upload_file
        delattr(hf_api, '_upload_file_origin')

def patch_hub():
    _patch_hub()
    _patch_pretrained_class()


def unpatch_hub():
    _unpatch_pretrained_class()
