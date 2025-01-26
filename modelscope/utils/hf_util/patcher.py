# Copyright (c) Alibaba, Inc. and its affiliates.
import contextlib
import importlib
import inspect
import os
import sys
from functools import partial
from pathlib import Path
from types import MethodType
from typing import BinaryIO, Dict, List, Optional, Union


def get_all_imported_modules():
    """Find all modules in transformers/peft/diffusers"""
    all_imported_modules = []
    transformers_include_names = [
        'Auto', 'T5', 'BitsAndBytes', 'GenerationConfig', 'Quant', 'Awq',
        'GPTQ', 'BatchFeature', 'Qwen2'
    ]
    diffusers_include_names = ['Pipeline']
    if importlib.util.find_spec('transformers') is not None:
        import transformers
        lazy_module = sys.modules['transformers']
        _import_structure = lazy_module._import_structure
        for key in _import_structure:
            values = _import_structure[key]
            for value in values:
                # pretrained
                if any([name in value for name in transformers_include_names]):
                    try:
                        module = importlib.import_module(
                            f'.{key}', transformers.__name__)
                        value = getattr(module, value)
                        all_imported_modules.append(value)
                    except (ImportError, AttributeError):
                        pass

    if importlib.util.find_spec('peft') is not None:
        import peft
        attributes = dir(peft)
        imports = [attr for attr in attributes if not attr.startswith('__')]
        all_imported_modules.extend(
            [getattr(peft, _import) for _import in imports])

    if importlib.util.find_spec('diffusers') is not None:
        import diffusers
        if importlib.util.find_spec('diffusers') is not None:
            lazy_module = sys.modules['diffusers']
            _import_structure = lazy_module._import_structure
            for key in _import_structure:
                values = _import_structure[key]
                for value in values:
                    if any([name in value
                            for name in diffusers_include_names]):
                        try:
                            module = importlib.import_module(
                                f'.{key}', diffusers.__name__)
                            value = getattr(module, value)
                            all_imported_modules.append(value)
                        except (ImportError, AttributeError):
                            pass
    return all_imported_modules


def _patch_pretrained_class(all_imported_modules, wrap=False):
    """Patch all class to download from modelscope

    Args:
        wrap: Wrap the class or monkey patch the original class

    Returns:
        The classes after patched
    """

    def get_model_dir(pretrained_model_name_or_path,
                      ignore_file_pattern=None,
                      allow_file_pattern=None,
                      **kwargs):
        from modelscope import snapshot_download
        if not os.path.exists(pretrained_model_name_or_path):
            model_dir = snapshot_download(
                pretrained_model_name_or_path,
                revision=kwargs.pop('revision', None),
                ignore_file_pattern=ignore_file_pattern,
                allow_file_pattern=allow_file_pattern)
        else:
            model_dir = pretrained_model_name_or_path
        return model_dir

    ignore_file_pattern = [
        r'\w+\.bin', r'\w+\.safetensors', r'\w+\.pth', r'\w+\.pt', r'\w+\.h5'
    ]

    def patch_pretrained_model_name_or_path(pretrained_model_name_or_path,
                                            *model_args, **kwargs):
        """Patch all from_pretrained/get_config_dict"""
        model_dir = get_model_dir(pretrained_model_name_or_path, **kwargs)
        return kwargs.pop('ori_func')(model_dir, *model_args, **kwargs)

    def patch_peft_model_id(model, model_id, *model_args, **kwargs):
        """Patch all peft.from_pretrained"""
        model_dir = get_model_dir(model_id, **kwargs)
        return kwargs.pop('ori_func')(model, model_dir, *model_args, **kwargs)

    def _get_peft_type(model_id, **kwargs):
        """Patch all _get_peft_type"""
        model_dir = get_model_dir(model_id, **kwargs)
        return kwargs.pop('ori_func')(model_dir, **kwargs)

    def get_wrapped_class(
            module_class: 'PreTrainedModel',
            ignore_file_pattern: Optional[Union[str, List[str]]] = None,
            allow_file_pattern: Optional[Union[str, List[str]]] = None,
            **kwargs):
        """Get a custom wrapper class for  auto classes to download the models from the ModelScope hub
        Args:
            module_class (`PreTrainedModel`): The actual module class
            ignore_file_pattern (`str` or `List`, *optional*, default to `None`):
                Any file pattern to be ignored, like exact file names or file extensions.
            allow_file_pattern (`str` or `List`, *optional*, default to `None`):
                Any file pattern to be included, like exact file names or file extensions.
        Returns:
            The wrapped class
        """

        def from_pretrained(model, model_id, *model_args, **kwargs):
            # model is an instance
            model_dir = get_model_dir(
                model_id,
                ignore_file_pattern=ignore_file_pattern,
                allow_file_pattern=allow_file_pattern,
                **kwargs)

            module_obj = module_class.from_pretrained(model, model_dir,
                                                      *model_args, **kwargs)

            return module_obj

        class ClassWrapper(module_class):

            @classmethod
            def from_pretrained(cls, pretrained_model_name_or_path,
                                *model_args, **kwargs):
                model_dir = get_model_dir(
                    pretrained_model_name_or_path,
                    ignore_file_pattern=ignore_file_pattern,
                    allow_file_pattern=allow_file_pattern,
                    **kwargs)

                module_obj = module_class.from_pretrained(
                    model_dir, *model_args, **kwargs)

                if module_class.__name__.startswith('AutoModel'):
                    module_obj.model_dir = model_dir
                return module_obj

            @classmethod
            def _get_peft_type(cls, model_id, **kwargs):
                model_dir = get_model_dir(model_id, **kwargs)
                module_obj = module_class._get_peft_type(model_dir, **kwargs)
                return module_obj

            @classmethod
            def get_config_dict(cls, pretrained_model_name_or_path,
                                *model_args, **kwargs):
                model_dir = get_model_dir(pretrained_model_name_or_path,
                                          **kwargs)

                module_obj = module_class.get_config_dict(
                    model_dir, *model_args, **kwargs)
                return module_obj

        if not hasattr(module_class, 'from_pretrained'):
            del ClassWrapper.from_pretrained
        else:
            parameters = inspect.signature(var.from_pretrained).parameters
            if 'model' in parameters and 'model_id' in parameters:
                # peft
                ClassWrapper.from_pretrained = from_pretrained

        if not hasattr(module_class, '_get_peft_type'):
            del ClassWrapper._get_peft_type

        if not hasattr(module_class, 'get_config_dict'):
            del ClassWrapper.get_config_dict

        ClassWrapper.__name__ = module_class.__name__
        ClassWrapper.__qualname__ = module_class.__qualname__
        return ClassWrapper

    all_available_modules = []
    for var in all_imported_modules:
        if var is None or not hasattr(var, '__name__'):
            continue
        name = var.__name__
        need_model = 'model' in name.lower() or 'processor' in name.lower(
        ) or 'extractor' in name.lower()
        if need_model:
            ignore_file_pattern_kwargs = {}
        else:
            ignore_file_pattern_kwargs = {
                'ignore_file_pattern': ignore_file_pattern
            }

        try:
            # some TFxxx classes has import errors
            has_from_pretrained = hasattr(var, 'from_pretrained')
            has_get_peft_type = hasattr(var, '_get_peft_type')
            has_get_config_dict = hasattr(var, 'get_config_dict')
        except ImportError:
            continue

        if wrap:
            try:
                if not has_from_pretrained and not has_get_config_dict and not has_get_peft_type:
                    all_available_modules.append(var)
                else:
                    all_available_modules.append(
                        get_wrapped_class(var, ignore_file_pattern))
            except Exception:
                all_available_modules.append(var)
        else:
            if has_from_pretrained and not hasattr(var,
                                                   '_from_pretrained_origin'):
                parameters = inspect.signature(var.from_pretrained).parameters
                # different argument names
                is_peft = 'model' in parameters and 'model_id' in parameters
                var._from_pretrained_origin = var.from_pretrained
                if not is_peft:
                    var.from_pretrained = partial(
                        patch_pretrained_model_name_or_path,
                        ori_func=var._from_pretrained_origin,
                        **ignore_file_pattern_kwargs)
                else:
                    var.from_pretrained = partial(
                        patch_peft_model_id,
                        ori_func=var._from_pretrained_origin,
                        **ignore_file_pattern_kwargs)
            if has_get_peft_type and not hasattr(var, '_get_peft_type_origin'):
                var._get_peft_type_origin = var._get_peft_type
                var._get_peft_type = partial(
                    _get_peft_type,
                    ori_func=var._get_peft_type_origin,
                    **ignore_file_pattern_kwargs)

            if has_get_config_dict and not hasattr(var,
                                                   '_get_config_dict_origin'):
                var._get_config_dict_origin = var.get_config_dict
                var.get_config_dict = partial(
                    patch_pretrained_model_name_or_path,
                    ori_func=var._get_config_dict_origin,
                    **ignore_file_pattern_kwargs)

            all_available_modules.append(var)
    return all_available_modules


def _unpatch_pretrained_class(all_imported_modules):
    for var in all_imported_modules:
        if var is None:
            continue

        try:
            has_from_pretrained = hasattr(var, 'from_pretrained')
            has_get_peft_type = hasattr(var, '_get_peft_type')
            has_get_config_dict = hasattr(var, 'get_config_dict')
        except ImportError:
            continue
        if has_from_pretrained and hasattr(var, '_from_pretrained_origin'):
            var.from_pretrained = var._from_pretrained_origin
            delattr(var, '_from_pretrained_origin')
        if has_get_peft_type and hasattr(var, '_get_peft_type_origin'):
            var._get_peft_type = var._get_peft_type_origin
            delattr(var, '_get_peft_type_origin')
        if has_get_config_dict and hasattr(var, '_get_config_dict_origin'):
            var.get_config_dict = var._get_config_dict_origin
            delattr(var, '_get_config_dict_origin')


def _patch_hub():
    import huggingface_hub
    from huggingface_hub import hf_api
    from huggingface_hub.hf_api import api
    from huggingface_hub.hf_api import future_compatible
    from modelscope import get_logger
    logger = get_logger()

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
            file_path=os.path.join(subfolder, filename)
            if subfolder else filename,
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
        self,
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
        from modelscope.hub.push_to_hub import push_files_to_hub
        push_files_to_hub(
            path_or_fileobj=folder_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            commit_message=commit_message,
            commit_description=commit_description,
            revision=revision,
            token=token)
        from modelscope.utils.repo_utils import CommitInfo
        return CommitInfo(
            commit_url=f'https://www.modelscope.cn/models/{repo_id}/files',
            commit_message=commit_message,
            commit_description=commit_description,
            oid=None,
        )

    from modelscope.utils.constant import DEFAULT_REPOSITORY_REVISION

    @future_compatible
    def upload_file(
        self,
        *,
        path_or_fileobj: Union[str, Path, bytes, BinaryIO],
        path_in_repo: str,
        repo_id: str,
        token: Union[str, bool, None] = None,
        revision: Optional[str] = DEFAULT_REPOSITORY_REVISION,
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
    _patch_pretrained_class(get_all_imported_modules())


def unpatch_hub():
    _unpatch_pretrained_class(get_all_imported_modules())
    _unpatch_hub()


@contextlib.contextmanager
def patch_context():
    patch_hub()
    yield
    unpatch_hub()
