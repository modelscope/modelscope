# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import os
import sys
from functools import partial
from pathlib import Path
import importlib
from types import MethodType
from typing import BinaryIO, Dict, List, Optional, Union

from huggingface_hub.hf_api import CommitInfo, future_compatible
from modelscope import snapshot_download
from modelscope.utils.constant import Invoke
from modelscope.utils.logger import get_logger


logger = get_logger()


extra_modules = ['T5']
lazy_module = sys.modules['transformers']
all_modules = lazy_module._modules
all_imported_modules = []
for module in all_modules:
    if 'auto' in module.lower() or any(m in module for m in extra_modules):
        all_imported_modules.append(importlib.import_module(f'transformers.{module}'))


def user_agent(invoked_by=None):
    if invoked_by is None:
        invoked_by = Invoke.PRETRAINED
    uagent = '%s/%s' % (Invoke.KEY, invoked_by)
    return uagent


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

        if name.endswith('HF'):
            has_from_pretrained = hasattr(var, 'from_pretrained')
            has_get_peft_type = hasattr(var, '_get_peft_type')
            parameters = inspect.signature(var.from_pretrained).parameters
            is_peft = 'model' in parameters and 'model_id' in parameters
            if has_from_pretrained:
                if not is_peft:
                    var.from_pretrained = partial(patch_pretrained_model_name_or_path,
                                                  ori_func=var.from_pretrained,
                                                  **ignore_file_pattern_kwargs)
                else:
                    var.from_pretrained = partial(patch_peft_model_id,
                                                  ori_func=var.from_pretrained,
                                                  **ignore_file_pattern_kwargs)
            if has_get_peft_type:
                var._get_peft_type = partial(_get_peft_type,
                                              ori_func=var._get_peft_type,
                                              **ignore_file_pattern_kwargs)


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
        return {'name': ModelScopeConfig.get_user_info()[0] or 'unknown'}

    # Patch hf_hub_download
    huggingface_hub.hf_hub_download = _file_download
    huggingface_hub.file_download.hf_hub_download = _file_download

    # Patch file_exists
    hf_api.file_exists = MethodType(_file_exists, api)
    huggingface_hub.file_exists = hf_api.file_exists
    huggingface_hub.hf_api.file_exists = hf_api.file_exists

    # Patch whoami
    hf_api.whoami = MethodType(_whoami, api)
    huggingface_hub.whoami = hf_api.whoami
    huggingface_hub.hf_api.whoami = hf_api.whoami

    # Patch repocard.validate
    from huggingface_hub import repocard
    repocard.RepoCard.validate = lambda *args, **kwargs: None

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

    # Patch create_repo
    from transformers.utils import hub
    hf_api.create_repo = MethodType(create_repo, api)
    huggingface_hub.create_repo = hf_api.create_repo
    huggingface_hub.hf_api.create_repo = hf_api.create_repo
    hub.create_repo = create_repo

    # Patch upload_folder
    hf_api.upload_folder = MethodType(upload_folder, api)
    huggingface_hub.upload_folder = hf_api.upload_folder
    huggingface_hub.hf_api.upload_folder = hf_api.upload_folder

    # Patch upload_file
    hf_api.upload_file = MethodType(upload_file, api)
    huggingface_hub.upload_file = hf_api.upload_file
    huggingface_hub.hf_api.upload_file = hf_api.upload_file
    repocard.upload_file = hf_api.upload_file


def patch_hub():
    _patch_hub()
    _patch_pretrained_class()
