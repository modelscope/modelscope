# Copyright (c) Alibaba, Inc. and its affiliates.
import contextlib
import importlib
import inspect
import os
import re
import sys
from functools import partial
from pathlib import Path
from types import MethodType
from typing import List, Optional, Union

ignore_file_pattern = [
    r'*.bin',
    r'*.safetensors',
    r'*.pth',
    r'*.pt',
    r'*.h5',
    r'*.ckpt',
    r'*.zip',
    r'*.onnx',
    r'*.tar',
    r'*.gz',
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
        # Recent transformers expose the lazy-loading structure as
        # ``_import_structure``; fall back to ``dir()`` when it is absent or
        # its layout changes across versions (mirrors the diffusers branch).
        if hasattr(lazy_module, '_import_structure'):
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
                        except Exception:  # noqa
                            pass
        else:
            all_imported_modules.extend([
                getattr(lazy_module, _import) for _import in dir(lazy_module)
                if not _import.startswith('__') and any([
                    re.fullmatch(name, _import)
                    for name in transformers_include_names
                ]) and hasattr(
                    getattr(lazy_module, _import, None), 'from_pretrained')
            ])

    if importlib.util.find_spec('peft') is not None:
        try:
            import peft
        except Exception:  # noqa
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
        except Exception:  # noqa
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
                            except Exception:  # noqa
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


def _decide_allow_file_pattern(module_name, cls=None):
    extra_allow_file_pattern = None
    if 'GenerationConfig' in module_name:
        from transformers.utils import GENERATION_CONFIG_NAME
        extra_allow_file_pattern = [GENERATION_CONFIG_NAME, r'*.py']
    elif 'Config' in module_name:
        from transformers import CONFIG_NAME
        extra_allow_file_pattern = [CONFIG_NAME, r'*.py']
    elif 'Tokenizer' in module_name:
        extra_allow_file_pattern = list((
            cls.vocab_files_names.values()
        ) if cls is not None and hasattr(cls, 'vocab_files_names') else []) + [
            'chat_template.jinja', r'*.json', r'*.py', r'*.txt', r'*.model',
            r'*.tiktoken'
        ]  # noqa
    elif 'Processor' in module_name:
        extra_allow_file_pattern = [
            'chat_template.jinja', r'*.json', r'*.py', r'*.txt', r'*.model',
            r'*.tiktoken'
        ]
    return extra_allow_file_pattern


def _ms_revision(revision):
    """Translate an HF revision string into one ModelScope accepts."""
    return 'master' if revision in (None, 'main') else revision


def _ms_download_kwargs_from_hf(kwargs, revision=None):
    """Map transformers download kwargs onto ``snapshot_download`` arguments.

    Forwards ``local_files_only``, ``cache_dir``, string ``token``, and an
    optional revision (normalized via ``_ms_revision``).
    """
    download_kwargs = {
        'local_files_only': kwargs.get('local_files_only', False),
    }
    cache_dir = kwargs.get('cache_dir')
    if cache_dir is not None:
        download_kwargs['cache_dir'] = cache_dir
    token = kwargs.get('token')
    if isinstance(token, str):
        download_kwargs['token'] = token
    if revision is not None:
        download_kwargs['revision'] = _ms_revision(revision)
    return download_kwargs


def _get_class_from_dynamic_module(class_reference, *args, **kwargs):
    """Wrapper that redirects dynamic-module downloads to ModelScope.

    When a config's ``auto_map`` references another repo, transformers calls
    ``get_class_from_dynamic_module`` to fetch it.  This wrapper ensures that
    fetch goes through ModelScope instead of HuggingFace.

    Cross-repo ``auto_map`` entries use ``repo_id--module.Class``.  After
    ``snapshot_download``, the local cache path may itself contain ``--``
    (modelscope_hub 0.1.x layout: ``models/{owner}--{name}/snapshots/...``).
    Re-joining that path with ``--`` would make transformers'
    ``class_reference.split("--")`` raise ``ValueError``.  Instead, pass the
    local directory as ``pretrained_model_name_or_path`` and the bare
    ``module.Class`` as ``class_reference`` so transformers takes the
    ``os.path.isdir`` branch.
    """
    from transformers.dynamic_module_utils import origin_get_class_from_dynamic_module
    has_pretrained_arg = (
        'pretrained_model_name_or_path'
        in inspect.signature(origin_get_class_from_dynamic_module).parameters)
    # Resolve pretrained_model_name_or_path from kwargs or positional args.
    # ``args`` is a tuple; never mutate it in place.
    pretrained_in_kwargs = False
    pretrained_model_name_or_path = None
    if has_pretrained_arg:
        if 'pretrained_model_name_or_path' in kwargs:
            pretrained_model_name_or_path = kwargs[
                'pretrained_model_name_or_path']
            pretrained_in_kwargs = True
        elif args:
            pretrained_model_name_or_path = args[0]
    if (pretrained_model_name_or_path is not None
            and not os.path.exists(pretrained_model_name_or_path)):
        from modelscope import snapshot_download
        # Model weights/config: use ``revision`` (not ``code_revision``).
        downloaded_path = snapshot_download(
            pretrained_model_name_or_path,
            **_ms_download_kwargs_from_hf(
                kwargs, revision=kwargs.get('revision')))
        if pretrained_in_kwargs:
            kwargs['pretrained_model_name_or_path'] = downloaded_path
        else:
            args = (downloaded_path, ) + args[1:]
        pretrained_model_name_or_path = downloaded_path
    if '--' in class_reference:
        # Only the first ``--`` is the auto_map delimiter (repo vs module).
        repo_id, class_reference = class_reference.split('--', 1)
        if not os.path.exists(repo_id):
            # Cross-repo code: transformers uses ``code_revision`` for this repo.
            download_kwargs = _ms_download_kwargs_from_hf(
                kwargs, revision=kwargs.get('code_revision'))
            extra_allow_file_pattern = _decide_allow_file_pattern(
                class_reference)
            if extra_allow_file_pattern is not None:
                download_kwargs[
                    'allow_file_pattern'] = extra_allow_file_pattern
            if 'Config' in class_reference or 'Processor' in class_reference or 'Tokenizer' in class_reference:
                download_kwargs['ignore_file_pattern'] = ignore_file_pattern
            from modelscope import snapshot_download
            repo_id = snapshot_download(repo_id, **download_kwargs)
        if has_pretrained_arg:
            # Local path + bare class name; do not rejoin with ``--``.
            # Keep kwargs/positional form consistent with the original call.
            if pretrained_in_kwargs:
                kwargs['pretrained_model_name_or_path'] = repo_id
            else:
                args = (repo_id, ) + args[1:]
        else:
            # Legacy transformers without pretrained_model_name_or_path.
            # Unsafe if repo_id (local cache) contains '--'; modern
            # transformers always take the branch above.
            class_reference = repo_id + '--' + class_reference
    return origin_get_class_from_dynamic_module(class_reference, *args,
                                                **kwargs)


def _patch_dynamic_module():
    """Globally patch ``get_class_from_dynamic_module`` to redirect to ModelScope.

    No-op when transformers is not installed, so ``patch_hub()`` stays usable
    in a transformers-less environment.
    """
    try:
        from transformers import dynamic_module_utils
    except ImportError:
        return
    if not hasattr(dynamic_module_utils,
                   'origin_get_class_from_dynamic_module'):
        dynamic_module_utils.origin_get_class_from_dynamic_module = dynamic_module_utils.get_class_from_dynamic_module
        dynamic_module_utils.get_class_from_dynamic_module = _get_class_from_dynamic_module
        from transformers.models.auto import configuration_auto
        configuration_auto.get_class_from_dynamic_module = _get_class_from_dynamic_module


def _unpatch_dynamic_module():
    try:
        from transformers import dynamic_module_utils
    except ImportError:
        return
    if hasattr(dynamic_module_utils, 'origin_get_class_from_dynamic_module'):
        dynamic_module_utils.get_class_from_dynamic_module = dynamic_module_utils.origin_get_class_from_dynamic_module
        from transformers.models.auto import configuration_auto
        configuration_auto.get_class_from_dynamic_module = dynamic_module_utils.origin_get_class_from_dynamic_module
        delattr(dynamic_module_utils, 'origin_get_class_from_dynamic_module')


@contextlib.contextmanager
def _dynamic_module_patch_scope():
    """Temporarily patch ``get_class_from_dynamic_module`` for the duration of
    the ``with`` block, unless an outer patch (e.g. ``patch_hub()``) is already
    in effect.

    This ensures that ``auto_map`` references inside a config are resolved via
    ModelScope when loading through a modelscope-wrapped class, without
    polluting the global transformers state for unrelated callers.
    """
    from transformers import dynamic_module_utils
    if hasattr(dynamic_module_utils, 'origin_get_class_from_dynamic_module'):
        yield
        return
    _patch_dynamic_module()
    try:
        yield
    finally:
        _unpatch_dynamic_module()


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
            revision = _ms_revision(kwargs.pop('revision', None))
            if file_filter is not None:
                allow_file_pattern = file_filter
            download_kwargs = _ms_download_kwargs_from_hf(
                kwargs, revision=revision)
            model_dir = snapshot_download(
                pretrained_model_name_or_path,
                ignore_file_pattern=ignore_file_pattern,
                allow_file_pattern=allow_file_pattern,
                **download_kwargs)
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
                extra_allow_file_pattern = _decide_allow_file_pattern(
                    module_class.__name__, cls)
                kwargs['allow_file_pattern'] = extra_allow_file_pattern
            yield
            kwargs.pop('ignore_file_pattern', None)
            kwargs.pop('allow_file_pattern', None)

        def from_pretrained(model, model_id, *model_args, **kwargs):

            # file_pattern_context requires (kwargs, module_class, cls);
            # module_class doubles as cls for the peft wrap path.
            with file_pattern_context(kwargs, module_class, module_class):
                # model is an instance
                model_dir = get_model_dir(model_id, **kwargs)

            with _dynamic_module_patch_scope():
                module_obj = module_class.from_pretrained(
                    model, model_dir, *model_args, **kwargs)

            return module_obj

        class ClassWrapper(module_class):

            @classmethod
            def from_pretrained(cls, pretrained_model_name_or_path,
                                *model_args, **kwargs):
                with file_pattern_context(kwargs, module_class, cls):
                    model_dir = get_model_dir(pretrained_model_name_or_path,
                                              **kwargs)

                with _dynamic_module_patch_scope():
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

                with _dynamic_module_patch_scope():
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
            parameters = inspect.signature(
                module_class.from_pretrained).parameters
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
        except Exception:  # noqa
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
            except Exception:  # noqa
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

    # Only apply the global get_class_from_dynamic_module monkey-patch in
    # direct-patch mode (wrap=False).  In wrap mode the ClassWrapper scopes
    # the patch to each from_pretrained / get_config_dict call via
    # _dynamic_module_patch_scope(), so the global state stays clean for
    # unrelated ``from transformers import AutoConfig`` callers (issue #1751).
    if not wrap:
        _patch_dynamic_module()
    return all_available_modules


def _unpatch_pretrained_class(all_imported_modules):
    # The patcher captured `var.from_pretrained` via descriptor access, which
    # returns a bound method. Re-wrap as classmethod so subclasses still get
    # `cls` bound to themselves (not to the ancestor that was patched).
    def _restore(var, attr, origin_attr):
        origin = getattr(var, origin_attr)
        if isinstance(origin, MethodType):
            setattr(var, attr, classmethod(origin.__func__))
        else:
            setattr(var, attr, origin)
        try:
            delattr(var, origin_attr)
        except Exception:  # noqa
            pass

    for var in all_imported_modules:
        if var is None:
            continue

        try:
            has_from_pretrained = hasattr(var, 'from_pretrained')
            has_get_peft_type = hasattr(var, '_get_peft_type')
            has_get_config_dict = hasattr(var, 'get_config_dict')
        except Exception:  # noqa
            continue
        if has_from_pretrained and hasattr(var, '_from_pretrained_origin'):
            _restore(var, 'from_pretrained', '_from_pretrained_origin')
        if has_get_peft_type and hasattr(var, '_get_peft_type_origin'):
            _restore(var, '_get_peft_type', '_get_peft_type_origin')
        if has_get_config_dict and hasattr(var, '_get_config_dict_origin'):
            _restore(var, 'get_config_dict', '_get_config_dict_origin')

    _unpatch_dynamic_module()


def _patch_kernels():
    """Monkey-patch the `kernels` library to route HF API calls to ModelScope.

    The implementation lives in `modelscope_hub.kernels`, so this package and
    a direct ``from modelscope_hub.kernels import get_kernel`` share one shim
    backed by the current hub SDK: one ``MsKernelApi``, one patch state.
    Requires ``modelscope-hub>=0.7.0``; no-op on older versions.
    """
    if not _require_hub_module(_KERNELS_COMPAT):
        return
    from modelscope_hub.kernels import patch_kernels_hub

    patch_kernels_hub()


def _unpatch_kernels():
    """Undo `_patch_kernels` via `modelscope_hub.kernels`."""
    if _hub_module_available(_KERNELS_COMPAT):
        from modelscope_hub.kernels import unpatch_kernels_hub

        unpatch_kernels_hub()


@contextlib.contextmanager
def _kernels_patch_scope():
    """Apply the kernels ModelScope patch for the duration of the `with`
    block, unless an outer patch (e.g. `patch_hub()`) is already in effect.

    Backed by `modelscope_hub.kernels.patch_scope`, so the global patch
    (`patch_hub`) and per-call scoped patches share one state. On
    ``modelscope-hub<0.7.0`` the scope is a no-op and `kernels` keeps using
    the Hugging Face Hub.
    """
    if not _hub_module_available(_KERNELS_COMPAT):
        yield
        return
    from modelscope_hub.kernels import patch_scope

    with patch_scope():
        yield


def _wrap_kernels_callable(attr_name):
    """Return a wrapper around `kernels.<attr_name>` that scopes the ModelScope
    patch on `kernels._get_hf_api` to the call itself, so
    `from kernels import <attr_name>` stays on HuggingFace unless the user
    explicitly calls `patch_hub()` / `patch_context()`.
    """

    def _wrapped(*args, **kwargs):
        import kernels
        with _kernels_patch_scope():
            return getattr(kernels, attr_name)(*args, **kwargs)

    _wrapped.__name__ = _wrapped.__qualname__ = attr_name
    return _wrapped


_HF_HUB_PATCHER = 'modelscope_hub.compat.hf_hub_patcher'
_KERNELS_COMPAT = 'modelscope_hub.kernels'
_warned_hub_070 = False


def _hub_module_available(module_name):
    """Whether the installed modelscope-hub provides ``module_name``.

    Hub-side huggingface_hub/kernels patching ships in
    ``modelscope-hub>=0.7.0``; on older versions the shims degrade to no-ops.
    """
    try:
        from importlib.util import find_spec
        return find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _require_hub_module(module_name):
    """Like `_hub_module_available`, but warns once when the module is absent."""
    global _warned_hub_070
    if _hub_module_available(module_name):
        return True
    if not _warned_hub_070:
        _warned_hub_070 = True
        import warnings
        warnings.warn(
            'The installed modelscope-hub is older than 0.7.0; '
            'huggingface_hub/kernels redirection is disabled until it is '
            'upgraded (requires modelscope-hub>=0.7.0).', stacklevel=3)
    return False


def _patch_hub():
    """Route huggingface_hub module functions to ModelScope via modelscope_hub.

    The implementation lives in ``modelscope_hub.compat.hf_hub_patcher``;
    this shim additionally bridges ``transformers.utils.hub`` (the
    transformers-side entrypoints of ``push_to_hub``) onto the patched
    huggingface_hub functions. No-op on ``modelscope-hub<0.7.0``.
    """
    if not _require_hub_module(_HF_HUB_PATCHER):
        return
    from modelscope_hub.compat.hf_hub_patcher import patch_hf_hub

    patch_hf_hub()
    _patch_transformers_hub()


def _unpatch_hub():
    """Undo `_patch_hub` (transformers bridge first, then the hub patches)."""
    _unpatch_transformers_hub()
    if _hub_module_available(_HF_HUB_PATCHER):
        from modelscope_hub.compat.hf_hub_patcher import unpatch_hf_hub

        unpatch_hf_hub()


def _patch_transformers_hub():
    """Point ``transformers.utils.hub`` at the patched huggingface_hub functions.

    Older transformers resolve ``create_repo`` / ``create_commit`` through
    ``transformers.utils.hub`` for ``push_to_hub``; redirect them to the
    already-patched ``huggingface_hub`` objects. Attributes absent in newer
    transformers are skipped instead of injected, and the whole bridge is
    skipped when transformers is not installed, so ``patch_hub()`` stays
    usable across versions and in a transformers-less env.
    """
    try:
        from transformers.utils import hub
    except ImportError:
        return
    import huggingface_hub

    if (hasattr(hub, 'create_repo')
            and not hasattr(hub, '_ms_origin_create_repo')):
        hub._ms_origin_create_repo = hub.create_repo
        hub.create_repo = huggingface_hub.create_repo
    if (hasattr(hub, 'create_commit')
            and not hasattr(hub, '_ms_origin_create_commit')):
        hub._ms_origin_create_commit = hub.create_commit
        hub.create_commit = huggingface_hub.create_commit


def _unpatch_transformers_hub():
    """Undo `_patch_transformers_hub` and restore ``transformers.utils.hub``."""
    try:
        from transformers.utils import hub
    except ImportError:
        return
    if hasattr(hub, '_ms_origin_create_repo'):
        hub.create_repo = hub._ms_origin_create_repo
        delattr(hub, '_ms_origin_create_repo')
    if hasattr(hub, '_ms_origin_create_commit'):
        hub.create_commit = hub._ms_origin_create_commit
        delattr(hub, '_ms_origin_create_commit')


# Modules captured by the last ``patch_hub()`` call. ``unpatch_hub()`` restores
# exactly these instead of re-enumerating: classes imported between patch and
# unpatch would otherwise be missed (or wrongly restored) by a fresh scan.
_patched_modules = None


def patch_hub():
    global _patched_modules
    _patch_hub()
    _patch_kernels()
    _patched_modules = get_all_imported_modules()
    _patch_pretrained_class(_patched_modules)


def unpatch_hub():
    global _patched_modules
    if _patched_modules is not None:
        _unpatch_pretrained_class(_patched_modules)
        _patched_modules = None
    _unpatch_kernels()
    _unpatch_hub()


@contextlib.contextmanager
def patch_context():
    patch_hub()
    try:
        yield
    finally:
        unpatch_hub()
