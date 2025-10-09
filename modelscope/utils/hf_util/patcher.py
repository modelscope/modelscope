# Copyright (c) Alibaba, Inc. and its affiliates.
import contextlib
import importlib
import inspect
import os
import re
import sys
from asyncio import Future
from functools import partial
from pathlib import Path
from types import MethodType
from typing import BinaryIO, Dict, Iterable, List, Optional, Union

from modelscope.hub.constants import DEFAULT_MODELSCOPE_DATA_ENDPOINT
from modelscope.utils.repo_utils import (CommitInfo, CommitOperation,
                                         CommitOperationAdd)

ignore_file_pattern = [
    r'\w+\.bin',
    r'\w+\.safetensors',
    r'\w+\.pth',
    r'\w+\.pt',
    r'\w+\.h5',
    r'\w+\.ckpt',
    r'\w+\.zip',
    r'\w+\.onnx',
    r'\w+\.tar',
    r'\w+\.gz',
]


def get_all_imported_modules():
    """Find all modules in transformers/peft/diffusers"""
    all_imported_modules = []
    transformers_include_names = [
        'Auto.*',
        'T5.*',
        'BitsAndBytesConfig',
        'GenerationConfig',
        'Awq.*',
        'GPTQ.*',
        'BatchFeature',
        'Qwen.*',
        'Llama.*',
        'Intern.*',
        'Deepseek.*',
        'PretrainedConfig',
        'PreTrainedTokenizer',
        'PreTrainedModel',
        'PreTrainedTokenizerFast',
    ]
    peft_include_names = ['.*PeftModel.*', '.*Config']
    diffusers_include_names = [
        '^(?!TF|Flax).*Pipeline$', '^(?!TF|Flax).*Autoencoder.*',
        '^(?!TF|Flax).*Model$', '^(?!TF|Flax).*Adapter$', 'ImageProjection',
        '^(?!TF|Flax).*UNet$', '^(?!TF|Flax).*Scheduler$'
    ]
    if importlib.util.find_spec('transformers') is not None:
        import transformers
        lazy_module = sys.modules['transformers']
        _import_structure = lazy_module._import_structure
        for key in _import_structure:
            if 'dummy' in key.lower():
                continue
            values = _import_structure[key]
            for value in values:
                # pretrained
                if any([
                        re.fullmatch(name, value)
                        for name in transformers_include_names
                ]):
                    try:
                        module = importlib.import_module(
                            f'.{key}', transformers.__name__)
                        value = getattr(module, value)
                        all_imported_modules.append(value)
                    except:  # noqa
                        pass

    if importlib.util.find_spec('peft') is not None:
        try:
            import peft
        except:  # noqa
            pass
        else:
            attributes = dir(peft)
            imports = [
                attr for attr in attributes if not attr.startswith('__')
            ]
            all_imported_modules.extend([
                getattr(peft, _import) for _import in imports if any([
                    re.fullmatch(name, _import) for name in peft_include_names
                ])
            ])

    if importlib.util.find_spec('diffusers') is not None:
        try:
            import diffusers
        except:  # noqa
            pass
        else:
            lazy_module = sys.modules['diffusers']
            if hasattr(lazy_module, '_import_structure'):
                _import_structure = lazy_module._import_structure
                for key in _import_structure:
                    if 'dummy' in key.lower():
                        continue
                    values = _import_structure[key]
                    for value in values:
                        if any([
                                re.fullmatch(name, value)
                                for name in diffusers_include_names
                        ]):
                            try:
                                module = importlib.import_module(
                                    f'.{key}', diffusers.__name__)
                                value = getattr(module, value)
                                all_imported_modules.append(value)
                            except:  # noqa
                                pass
            else:
                attributes = dir(lazy_module)
                imports = [
                    attr for attr in attributes if not attr.startswith('__')
                ]
                all_imported_modules.extend([
                    getattr(lazy_module, _import) for _import in imports
                    if any([
                        re.fullmatch(name, _import)
                        for name in diffusers_include_names
                    ])
                ])
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
        subfolder = kwargs.pop('subfolder', None)
        file_filter = None
        if subfolder:
            file_filter = f'{subfolder}/*'
        if not os.path.exists(pretrained_model_name_or_path):
            revision = kwargs.pop('revision', None)
            if revision is None or revision == 'main':
                revision = 'master'
            if file_filter is not None:
                allow_file_pattern = file_filter
            model_dir = snapshot_download(
                pretrained_model_name_or_path,
                revision=revision,
                ignore_file_pattern=ignore_file_pattern,
                allow_file_pattern=allow_file_pattern)
            if subfolder:
                model_dir = os.path.join(model_dir, subfolder)
        else:
            model_dir = pretrained_model_name_or_path
        return model_dir

    def patch_pretrained_model_name_or_path(cls, pretrained_model_name_or_path,
                                            *model_args, **kwargs):
        """Patch all from_pretrained"""
        model_dir = get_model_dir(pretrained_model_name_or_path,
                                  kwargs.pop('ignore_file_pattern', None),
                                  kwargs.pop('allow_file_pattern', None),
                                  **kwargs)
        return cls._from_pretrained_origin.__func__(cls, model_dir,
                                                    *model_args, **kwargs)

    def patch_get_config_dict(cls, pretrained_model_name_or_path, *model_args,
                              **kwargs):
        """Patch all get_config_dict"""
        model_dir = get_model_dir(pretrained_model_name_or_path,
                                  kwargs.pop('ignore_file_pattern', None),
                                  kwargs.pop('allow_file_pattern', None),
                                  **kwargs)
        return cls._get_config_dict_origin.__func__(cls, model_dir,
                                                    *model_args, **kwargs)

    def patch_peft_model_id(cls, model, model_id, *model_args, **kwargs):
        """Patch all peft.from_pretrained"""
        model_dir = get_model_dir(model_id,
                                  kwargs.pop('ignore_file_pattern', None),
                                  kwargs.pop('allow_file_pattern', None),
                                  **kwargs)
        return cls._from_pretrained_origin.__func__(cls, model, model_dir,
                                                    *model_args, **kwargs)

    def patch_get_peft_type(cls, model_id, **kwargs):
        """Patch all _get_peft_type"""
        model_dir = get_model_dir(model_id,
                                  kwargs.pop('ignore_file_pattern', None),
                                  kwargs.pop('allow_file_pattern', None),
                                  **kwargs)
        return cls._get_peft_type_origin.__func__(cls, model_dir, **kwargs)

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

        @contextlib.contextmanager
        def file_pattern_context(kwargs, module_class, cls):
            if 'allow_file_pattern' not in kwargs:
                kwargs['allow_file_pattern'] = allow_file_pattern
            if 'ignore_file_pattern' not in kwargs:
                kwargs['ignore_file_pattern'] = ignore_file_pattern

            if kwargs.get(
                    'allow_file_pattern') is None and module_class is not None:
                extra_allow_file_pattern = None
                if 'GenerationConfig' == module_class.__name__:
                    from transformers.utils import GENERATION_CONFIG_NAME
                    extra_allow_file_pattern = [
                        GENERATION_CONFIG_NAME, r'*.py'
                    ]
                elif 'Config' in module_class.__name__:
                    from transformers import CONFIG_NAME
                    extra_allow_file_pattern = [CONFIG_NAME, r'*.py']
                elif 'Tokenizer' in module_class.__name__:
                    extra_allow_file_pattern = list(
                        (cls.vocab_files_names.values()) if cls is not None
                        and hasattr(cls, 'vocab_files_names') else []) + [
                            'chat_template.jinja', r'*.json', r'*.py',
                            r'*.txt', r'*.model', r'*.tiktoken'
                        ]  # noqa
                elif 'Processor' in module_class.__name__:
                    extra_allow_file_pattern = [
                        'chat_template.jinja', r'*.json', r'*.py', r'*.txt',
                        r'*.model', r'*.tiktoken'
                    ]

                kwargs['allow_file_pattern'] = extra_allow_file_pattern
            yield
            kwargs.pop('ignore_file_pattern', None)
            kwargs.pop('allow_file_pattern', None)

        def from_pretrained(model, model_id, *model_args, **kwargs):

            with file_pattern_context(kwargs):
                # model is an instance
                model_dir = get_model_dir(
                    model_id,
                    module_class=module_class,
                    cls=module_class,
                    **kwargs)

            module_obj = module_class.from_pretrained(model, model_dir,
                                                      *model_args, **kwargs)

            return module_obj

        class ClassWrapper(module_class):

            @classmethod
            def from_pretrained(cls, pretrained_model_name_or_path,
                                *model_args, **kwargs):
                with file_pattern_context(kwargs, module_class, cls):
                    model_dir = get_model_dir(pretrained_model_name_or_path,
                                              **kwargs)

                module_obj = module_class.from_pretrained(
                    model_dir, *model_args, **kwargs)

                if module_class.__name__.startswith('AutoModel'):
                    module_obj.model_dir = model_dir
                return module_obj

            @classmethod
            def _get_peft_type(cls, model_id, **kwargs):
                with file_pattern_context(kwargs, module_class, cls):
                    model_dir = get_model_dir(
                        model_id,
                        ignore_file_pattern=ignore_file_pattern,
                        allow_file_pattern=allow_file_pattern,
                        **kwargs)

                module_obj = module_class._get_peft_type(model_dir, **kwargs)
                return module_obj

            @classmethod
            def get_config_dict(cls, pretrained_model_name_or_path,
                                *model_args, **kwargs):
                with file_pattern_context(kwargs, module_class, cls):
                    model_dir = get_model_dir(
                        pretrained_model_name_or_path,
                        ignore_file_pattern=ignore_file_pattern,
                        allow_file_pattern=allow_file_pattern,
                        **kwargs)

                module_obj = module_class.get_config_dict(
                    model_dir, *model_args, **kwargs)
                return module_obj

            def save_pretrained(
                self,
                save_directory: Union[str, os.PathLike],
                safe_serialization: bool = True,
                **kwargs,
            ):
                push_to_hub = kwargs.pop('push_to_hub', False)
                if push_to_hub:
                    from modelscope.hub.push_to_hub import push_to_hub
                    from modelscope.hub.api import HubApi
                    from modelscope.hub.repository import Repository

                    token = kwargs.get('token')
                    commit_message = kwargs.pop('commit_message', None)
                    repo_name = kwargs.pop(
                        'repo_id',
                        save_directory.split(os.path.sep)[-1])

                    api = HubApi()
                    api.login(token)
                    api.create_repo(repo_name)
                    # clone the repo
                    Repository(save_directory, repo_name)

                super().save_pretrained(
                    save_directory=save_directory,
                    safe_serialization=safe_serialization,
                    push_to_hub=False,
                    **kwargs)

                # Class members may be unpatched, so push_to_hub is done separately here
                if push_to_hub:
                    push_to_hub(
                        repo_name=repo_name,
                        output_dir=save_directory,
                        commit_message=commit_message,
                        token=token)

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

        if not hasattr(module_class, 'save_pretrained'):
            del ClassWrapper.save_pretrained

        ClassWrapper.__name__ = module_class.__name__
        ClassWrapper.__qualname__ = module_class.__qualname__
        return ClassWrapper

    all_available_modules = []
    for var in all_imported_modules:
        if var is None or not hasattr(var, '__name__'):
            continue
        name = var.__name__
        skip_model = 'tokenizer' in name.lower() or 'config' in name.lower()
        if not skip_model:
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
            has_save_pretrained = hasattr(var, 'save_pretrained')
        except:  # noqa
            continue

        # save_pretrained is not a classmethod and cannot be overridden by replacing
        # the class method. It requires replacing the class object method.
        if wrap or ('pipeline' in name.lower() and has_save_pretrained):
            try:
                if (not has_from_pretrained and not has_get_config_dict
                        and not has_get_peft_type and not has_save_pretrained):
                    all_available_modules.append(var)
                else:
                    all_available_modules.append(
                        get_wrapped_class(var, **ignore_file_pattern_kwargs))
            except:  # noqa
                all_available_modules.append(var)
        else:
            if has_from_pretrained and not hasattr(var,
                                                   '_from_pretrained_origin'):
                parameters = inspect.signature(var.from_pretrained).parameters
                # different argument names
                is_peft = 'model' in parameters and 'model_id' in parameters
                var._from_pretrained_origin = var.from_pretrained
                if not is_peft:
                    var.from_pretrained = classmethod(
                        partial(patch_pretrained_model_name_or_path,
                                **ignore_file_pattern_kwargs))
                else:
                    var.from_pretrained = classmethod(
                        partial(patch_peft_model_id,
                                **ignore_file_pattern_kwargs))
            if has_get_peft_type and not hasattr(var, '_get_peft_type_origin'):
                var._get_peft_type_origin = var._get_peft_type
                var._get_peft_type = classmethod(
                    partial(patch_get_peft_type, **ignore_file_pattern_kwargs))

            if has_get_config_dict and not hasattr(var,
                                                   '_get_config_dict_origin'):
                var._get_config_dict_origin = var.get_config_dict
                var.get_config_dict = classmethod(
                    partial(patch_get_config_dict,
                            **ignore_file_pattern_kwargs))

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
        except:  # noqa
            continue
        if has_from_pretrained and hasattr(var, '_from_pretrained_origin'):
            var.from_pretrained = var._from_pretrained_origin
            try:
                delattr(var, '_from_pretrained_origin')
            except:  # noqa
                pass
        if has_get_peft_type and hasattr(var, '_get_peft_type_origin'):
            var._get_peft_type = var._get_peft_type_origin
            try:
                delattr(var, '_get_peft_type_origin')
            except:  # noqa
                pass
        if has_get_config_dict and hasattr(var, '_get_config_dict_origin'):
            var.get_config_dict = var._get_config_dict_origin
            try:
                delattr(var, '_get_config_dict_origin')
            except:  # noqa
                pass


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
        api.login(token)
        if revision is None or revision == 'main':
            revision = 'master'
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
        api.login(token)
        if revision is None or revision == 'main':
            revision = 'master'
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
        api.login(token)
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
        from modelscope.hub.api import HubApi
        api = HubApi()
        visibility = 'private' if private else 'public'
        repo_url = api.create_repo(
            repo_id, token=token, visibility=visibility, **kwargs)
        from modelscope.utils.repo_utils import RepoUrl
        return RepoUrl(url=repo_url, repo_type='model', repo_id=repo_id)

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
        from modelscope.hub.push_to_hub import _push_files_to_hub
        if revision is None or revision == 'main':
            revision = 'master'
        _push_files_to_hub(
            path_or_fileobj=folder_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            commit_message=commit_message,
            commit_description=commit_description,
            revision=revision,
            token=token)
        from modelscope.utils.repo_utils import CommitInfo
        return CommitInfo(
            commit_url=
            f'{DEFAULT_MODELSCOPE_DATA_ENDPOINT}/models/{repo_id}/files',
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
        if revision is None or revision == 'main':
            revision = 'master'
        from modelscope.hub.push_to_hub import _push_files_to_hub
        _push_files_to_hub(path_or_fileobj, path_in_repo, repo_id, token,
                           revision, commit_message, commit_description)

    @future_compatible
    def create_commit(
        self,
        repo_id: str,
        operations: Iterable[CommitOperation],
        *,
        commit_message: str,
        commit_description: Optional[str] = None,
        token: Union[str, bool, None] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = DEFAULT_REPOSITORY_REVISION,
        **kwargs,
    ) -> Union[CommitInfo, Future[CommitInfo]]:
        from modelscope.hub.api import HubApi
        api = HubApi()
        if any(['Add' not in op.__class__.__name__ for op in operations]):
            raise ValueError(
                'ModelScope create_commit only support Add operation for now.')
        if revision is None or revision == 'main':
            revision = 'master'
        all_files = [op.path_or_fileobj for op in operations]
        api.upload_folder(
            repo_id=repo_id,
            folder_path=all_files,
            commit_message=commit_message,
            commit_description=commit_description,
            token=token,
            revision=revision,
            repo_type=repo_type or 'model')

    def load(
        cls,
        repo_id_or_path: Union[str, Path],
        repo_type: Optional[str] = None,
        token: Optional[str] = None,
        ignore_metadata_errors: bool = False,
    ):
        from modelscope.hub.api import HubApi
        api = HubApi()
        api.login(token)
        if os.path.exists(repo_id_or_path):
            file_path = repo_id_or_path
        elif repo_type == 'model' or repo_type is None:
            from modelscope import model_file_download
            file_path = model_file_download(repo_id_or_path, 'README.md')
        elif repo_type == 'dataset':
            from modelscope import dataset_file_download
            file_path = dataset_file_download(repo_id_or_path, 'README.md')
        else:
            raise ValueError(
                f'repo_type should be `model` or `dataset`, but now is {repo_type}'
            )

        with open(file_path, 'r') as f:
            repo_card = cls(
                f.read(), ignore_metadata_errors=ignore_metadata_errors)
            if not hasattr(repo_card.data, 'tags'):
                repo_card.data.tags = []
            return repo_card

    # Patch repocard.validate
    from huggingface_hub import repocard
    if not hasattr(repocard.RepoCard, '_validate_origin'):
        repocard.RepoCard._validate_origin = repocard.RepoCard.validate
        repocard.RepoCard.validate = lambda *args, **kwargs: None
        repocard.RepoCard._load_origin = repocard.RepoCard.load
        repocard.RepoCard.load = MethodType(load, repocard.RepoCard)

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

    if not hasattr(hf_api, '_create_commit_origin'):
        # Patch upload_file
        hf_api._create_commit_origin = hf_api.create_commit
        hf_api.create_commit = MethodType(create_commit, api)
        huggingface_hub.create_commit = hf_api.create_commit
        huggingface_hub.hf_api.create_commit = hf_api.create_commit
        from transformers.utils import hub
        hub.create_commit = hf_api.create_commit


def _unpatch_hub():
    import huggingface_hub
    from huggingface_hub import hf_api

    from huggingface_hub import repocard
    if hasattr(repocard.RepoCard, '_validate_origin'):
        repocard.RepoCard.validate = repocard.RepoCard._validate_origin
        delattr(repocard.RepoCard, '_validate_origin')
    if hasattr(repocard.RepoCard, '_load_origin'):
        repocard.RepoCard.load = repocard.RepoCard._load_origin
        delattr(repocard.RepoCard, '_load_origin')

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

    if hasattr(hf_api, '_create_commit_origin'):
        hf_api.create_commit = hf_api._create_commit_origin
        huggingface_hub.create_commit = hf_api.create_commit
        huggingface_hub.hf_api.create_commit = hf_api.create_commit
        from transformers.utils import hub
        hub.create_commit = hf_api.create_commit
        delattr(hf_api, '_create_commit_origin')


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
