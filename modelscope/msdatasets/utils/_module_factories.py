# Copyright (c) Alibaba, Inc. and its affiliates.
"""Dataset module factory functions and data file resolution for ModelScope.

This module provides ModelScope-specific implementations of dataset module
loading (both script-based and script-free) and data file pattern resolution.
These functions are monkey-patched onto the ``datasets`` library internals
by :func:`~hf_datasets_util.load_dataset_with_ctx`.
"""
import importlib
import inspect
import os
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from datasets import (BuilderConfig, DownloadConfig, DownloadMode, Features,
                      Version, config, data_files)
from datasets.data_files import (
    FILES_TO_IGNORE, DataFilesDict, EmptyDatasetError,
    _get_data_files_patterns, _is_inside_unrequested_special_dir,
    _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir,
    sanitize_patterns)
from datasets.download.streaming_download_manager import (
    _prepare_path_and_storage_options, xbasename, xjoin)
from datasets.exceptions import DataFilesNotFoundError
from datasets.info import DatasetInfosDict
from datasets.load import (BuilderConfigsParameters, DatasetModule,
                           create_builder_configs_from_metadata_configs,
                           get_dataset_builder_class, import_main_class,
                           infer_module_for_data_files)
from datasets.naming import camelcase_to_snakecase
from datasets.packaged_modules import (_MODULE_TO_EXTENSIONS,
                                       _PACKAGED_DATASETS_MODULES)
from datasets.utils.file_utils import (cached_path, is_local_path,
                                       relative_to_absolute_path)
from datasets.utils.metadata import MetadataConfigs
from datasets.utils.track import tracked_str
from fsspec import filesystem
from fsspec.core import _un_chain
from fsspec.utils import stringify_path
from huggingface_hub import DatasetCard, DatasetCardData
from packaging import version

from modelscope import HubApi
from modelscope.msdatasets.utils._compat import (
    _HAS_SCRIPT_LOADING, _create_importable_file, _get_importable_file_path,
    _load_importable_file, files_to_hash, get_imports, init_dynamic_modules,
    resolve_trust_remote_code)
from modelscope.utils.constant import (DEFAULT_DATASET_REVISION,
                                       REPO_TYPE_DATASET)
from modelscope.utils.file_utils import is_relative_path
from modelscope.utils.import_utils import has_attr_in_class
from modelscope.utils.logger import get_logger

# ALL_ALLOWED_EXTENSIONS moved to datasets.packaged_modules in datasets 4.0
try:
    from datasets.packaged_modules import _ALL_ALLOWED_EXTENSIONS as ALL_ALLOWED_EXTENSIONS
except ImportError:
    from datasets.load import ALL_ALLOWED_EXTENSIONS

logger = get_logger()

# ---------------------------------------------------------------------------
# Shared HubApi instance (avoids creating a new requests.Session per call)
# ---------------------------------------------------------------------------
_hub_api: Optional[HubApi] = None


def _get_hub_api() -> HubApi:
    global _hub_api
    if _hub_api is None:
        _hub_api = HubApi(timeout=3 * 60, max_retries=3)
    return _hub_api


# ===================================================================
# Data file resolution
# ===================================================================


def get_fs_token_paths(
    urlpath,
    storage_options=None,
    protocol=None,
):
    if isinstance(urlpath, (list, tuple, set)):
        if not urlpath:
            raise ValueError('empty urlpath sequence')
        urlpath0 = stringify_path(list(urlpath)[0])
    else:
        urlpath0 = stringify_path(urlpath)
    storage_options = storage_options or {}
    if protocol:
        storage_options['protocol'] = protocol
    chain = _un_chain(urlpath0, storage_options or {})
    inkwargs = {}
    for i, ch in enumerate(reversed(chain)):
        urls, nested_protocol, kw = ch
        if i == len(chain) - 1:
            inkwargs = dict(**kw, **inkwargs)
            continue
        inkwargs['target_options'] = dict(**kw, **inkwargs)
        inkwargs['target_protocol'] = nested_protocol
        inkwargs['fo'] = urls
    paths, protocol, _ = chain[0]
    fs = filesystem(protocol, **inkwargs)
    return fs


def _resolve_pattern(
    pattern: str,
    base_path: str,
    allowed_extensions: Optional[List[str]] = None,
    download_config: Optional[DownloadConfig] = None,
) -> List[str]:
    """Resolve data file paths/URLs from a user-supplied pattern.

    Supports ``*``, ``**``, and fsspec-based remote patterns (e.g. ``hf://``).
    Hidden files/directories and ``__pycache__`` are excluded by default.
    """
    if is_relative_path(pattern):
        pattern = xjoin(base_path, pattern)
    elif is_local_path(pattern):
        base_path = os.path.splitdrive(pattern)[0] + os.sep
    else:
        base_path = ''
    pattern, storage_options = _prepare_path_and_storage_options(
        pattern, download_config=download_config)
    fs = get_fs_token_paths(pattern, storage_options=storage_options)
    fs_base_path = base_path.split('::')[0].split('://')[-1] or fs.root_marker
    fs_pattern = pattern.split('::')[0].split('://')[-1]
    files_to_ignore = set(FILES_TO_IGNORE) - {xbasename(pattern)}
    protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
    protocol_prefix = protocol + '://' if protocol != 'file' else ''
    glob_kwargs = {}
    if protocol == 'hf' and config.HF_HUB_VERSION >= version.parse('0.20.0'):
        glob_kwargs['expand_info'] = False

    try:
        tmp_file_paths = fs.glob(pattern, detail=True, **glob_kwargs)
    except FileNotFoundError:
        raise DataFilesNotFoundError(f"Unable to find '{pattern}'")

    matched_paths = [
        filepath if filepath.startswith(protocol_prefix) else protocol_prefix
        + filepath for filepath, info in tmp_file_paths.items()
        if info['type'] == 'file' and (
            xbasename(filepath) not in files_to_ignore)
        and not _is_inside_unrequested_special_dir(
            os.path.relpath(filepath, fs_base_path),
            os.path.relpath(fs_pattern, fs_base_path)) and  # noqa: W504
        not _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(  # noqa: W504
            os.path.relpath(filepath, fs_base_path),
            os.path.relpath(fs_pattern, fs_base_path))
    ]
    if allowed_extensions is not None:
        out = [
            filepath for filepath in matched_paths
            if any('.' + suffix in allowed_extensions
                   for suffix in xbasename(filepath).split('.')[1:])
        ]
        if len(out) < len(matched_paths):
            invalid_matched_files = list(set(matched_paths) - set(out))
            logger.info(
                f"Some files matched the pattern '{pattern}' but don't have valid data file extensions: "
                f'{invalid_matched_files}')
    else:
        out = matched_paths
    if not out:
        error_msg = f"Unable to find '{pattern}'"
        if allowed_extensions is not None:
            error_msg += f' with any supported extension {list(allowed_extensions)}'
        raise FileNotFoundError(error_msg)
    return out


def _get_data_patterns(
        base_path: str,
        download_config: Optional[DownloadConfig] = None
) -> Dict[str, List[str]]:
    """Get data file patterns for a dataset directory.

    Tries ``SPLIT_PATTERN_SHARDED`` first, then falls back to
    ``ALL_DEFAULT_PATTERNS``.
    """
    resolver = partial(
        _resolve_pattern, base_path=base_path, download_config=download_config)
    try:
        return _get_data_files_patterns(resolver)
    except FileNotFoundError:
        raise EmptyDatasetError(
            f"The directory at {base_path} doesn't contain any data files"
        ) from None


# ===================================================================
# Repository file download helper
# ===================================================================


def _download_repo_file(
    repo_id: str,
    path_in_repo: str,
    download_config: DownloadConfig,
    revision: str,
) -> str:
    """Download a single file from a ModelScope dataset repository."""
    api = _get_hub_api()
    _namespace, _dataset_name = repo_id.split('/')
    endpoint = api.get_endpoint_for_read(
        repo_id=repo_id, repo_type=REPO_TYPE_DATASET)
    if download_config and download_config.download_desc is None:
        download_config.download_desc = f'Downloading [{path_in_repo}]'
    try:
        url_or_filename = api.get_dataset_file_url(
            file_name=path_in_repo,
            dataset_name=_dataset_name,
            namespace=_namespace,
            revision=revision,
            extension_filter=False,
            endpoint=endpoint,
        )
        repo_file_path = cached_path(
            url_or_filename=url_or_filename, download_config=download_config)
    except FileNotFoundError as e:
        repo_file_path = ''
        logger.error(e)
    return repo_file_path


# ===================================================================
# Additional modules download (for script-based datasets)
# ===================================================================


def _download_additional_modules(
    name: str,
    dataset_name: str,
    namespace: str,
    revision: str,
    imports: Tuple[str, str, str, str],
    download_config: Optional[DownloadConfig],
    trust_remote_code: Optional[bool] = False,
) -> List[Tuple[str, str]]:
    """Download additional modules referenced by a dataset builder script.

    Parses the import list produced by ``get_imports`` and downloads any
    internal (relative) or external modules. Library imports are validated
    but not downloaded.
    """
    local_imports: List[Tuple[str, str]] = []
    library_imports: List[Tuple[str, str]] = []

    has_remote_code = any(
        import_type in ('internal', 'external')
        for import_type, _, _, _ in imports)
    if has_remote_code and not trust_remote_code:
        raise ValueError(
            f'Loading {name} requires executing code from the repository. '
            'This is disabled by default for security reasons. '
            'If you trust the authors of this dataset, you can enable it with '
            '`trust_remote_code=True`.')

    api = _get_hub_api()
    download_config = download_config.copy()
    if download_config.download_desc is None:
        download_config.download_desc = 'Downloading extra modules'

    for import_type, import_name, import_path, sub_directory in imports:
        if import_type == 'library':
            library_imports.append((import_name, import_path))
            continue
        if import_name == name:
            raise ValueError(
                f'Error in the {name} script, importing relative {import_name} module '
                f'but {import_name} is the name of the script. '
                f"Please change relative import {import_name} to another name and add a '# From: URL_OR_PATH' "
                f'comment pointing to the original relative import file path.')
        if import_type == 'internal':
            file_name = import_path + '.py'
            url_or_filename = api.get_dataset_file_url(
                file_name=file_name,
                dataset_name=dataset_name,
                namespace=namespace,
                revision=revision,
            )
        elif import_type == 'external':
            url_or_filename = import_path
        else:
            raise ValueError('Wrong import_type')

        local_import_path = cached_path(
            url_or_filename, download_config=download_config)
        if sub_directory is not None:
            local_import_path = os.path.join(local_import_path, sub_directory)
        local_imports.append((import_name, local_import_path))

    # Validate library imports
    needs_to_be_installed = {}
    for library_import_name, library_import_path in library_imports:
        try:
            importlib.import_module(library_import_name)
        except ImportError:
            if library_import_name not in needs_to_be_installed or library_import_path != library_import_name:
                needs_to_be_installed[
                    library_import_name] = library_import_path
    if needs_to_be_installed:
        _dependencies_str = 'dependencies' if len(
            needs_to_be_installed) > 1 else 'dependency'
        _them_str = 'them' if len(needs_to_be_installed) > 1 else 'it'
        if 'sklearn' in needs_to_be_installed:
            needs_to_be_installed['sklearn'] = 'scikit-learn'
        if 'Bio' in needs_to_be_installed:
            needs_to_be_installed['Bio'] = 'biopython'
        raise ImportError(
            f'To be able to use {name}, you need to install the following {_dependencies_str}: '
            f"{', '.join(needs_to_be_installed)}.\nPlease install {_them_str} using 'pip install "
            f"{' '.join(needs_to_be_installed.values())}' for instance.")
    return local_imports


# ===================================================================
# Module factory: script-based (Hub)
# ===================================================================


def _load_script_module(
    repo_id: str,
    revision: str,
    download_config: DownloadConfig,
    download_mode=None,
    dynamic_modules_path: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
) -> DatasetModule:
    """Shared implementation for loading a dataset module from a Hub .py script.

    Used by both ``get_module_with_script`` (monkey-patch for datasets<4.0) and
    ``_compat_hub_script_module`` (compat shim for datasets>=4.0).
    """
    _namespace, _dataset_name = repo_id.split('/')
    script_file_name = f'{_dataset_name}.py'

    local_script_path = _download_repo_file(
        repo_id=repo_id,
        path_in_repo=script_file_name,
        download_config=download_config,
        revision=revision,
    )
    if not local_script_path:
        raise FileNotFoundError(
            f'Cannot find {script_file_name} in {repo_id} at revision {revision}.'
        )

    dataset_readme_path = _download_repo_file(
        repo_id=repo_id,
        path_in_repo='README.md',
        download_config=download_config,
        revision=revision,
    )

    imports = get_imports(local_script_path)
    local_imports = _download_additional_modules(
        name=repo_id,
        dataset_name=_dataset_name,
        namespace=_namespace,
        revision=revision,
        imports=imports,
        download_config=download_config,
        trust_remote_code=trust_remote_code,
    )

    additional_files = []
    if dataset_readme_path:
        additional_files.append(
            (config.REPOCARD_FILENAME, dataset_readme_path))

    dynamic_modules_path = dynamic_modules_path or init_dynamic_modules()
    hash_val = files_to_hash([local_script_path]
                             + [loc[1] for loc in local_imports])
    importable_file_path = _get_importable_file_path(
        dynamic_modules_path=dynamic_modules_path,
        module_namespace='datasets',
        subdirectory_name=hash_val,
        name=repo_id,
    )
    if not os.path.exists(importable_file_path):
        trust = resolve_trust_remote_code(
            trust_remote_code=trust_remote_code, repo_id=repo_id)
        if trust:
            logger.warning(
                f'Use trust_remote_code=True. Will invoke codes from {repo_id}. '
                'Please make sure that you can trust the external codes.')
            _create_importable_file(
                local_path=local_script_path,
                local_imports=local_imports,
                additional_files=additional_files,
                dynamic_modules_path=dynamic_modules_path,
                module_namespace='datasets',
                subdirectory_name=hash_val,
                name=repo_id,
                download_mode=download_mode,
            )
        else:
            raise ValueError(
                f'Loading {repo_id} requires executing the dataset script in that'
                ' repo on your local machine. Make sure you have read the code there to avoid malicious use, then'
                ' set the option `trust_remote_code=True` to remove this error.'
            )
    module_path, hash_val = _load_importable_file(
        dynamic_modules_path=dynamic_modules_path,
        module_namespace='datasets',
        subdirectory_name=hash_val,
        name=repo_id,
    )
    importlib.invalidate_caches()

    api = _get_hub_api()
    builder_kwargs = {
        'base_path': api.get_file_base_path(repo_id=repo_id),
        'repo_id': repo_id,
    }
    return DatasetModule(module_path, hash_val, builder_kwargs)


def get_module_with_script(self) -> DatasetModule:
    """Monkey-patch target for ``HubDatasetModuleFactoryWithScript.get_module`` (datasets<4.0)."""
    repo_id: str = self.name
    revision = self.download_config.storage_options.get(
        'revision', None) or DEFAULT_DATASET_REVISION
    return _load_script_module(
        repo_id=repo_id,
        revision=revision,
        download_config=self.download_config,
        download_mode=self.download_mode,
        dynamic_modules_path=self.dynamic_modules_path
        if self.dynamic_modules_path else None,
        trust_remote_code=self.trust_remote_code,
    )


def _compat_hub_script_module(
    path,
    revision=None,
    download_config=None,
    download_mode=None,
    dynamic_modules_path=None,
    trust_remote_code=None,
) -> DatasetModule:
    """Load a dataset module from a Hub repo .py script (compat for datasets>=4.0)."""
    return _load_script_module(
        repo_id=path,
        revision=revision or DEFAULT_DATASET_REVISION,
        download_config=download_config or DownloadConfig(),
        download_mode=download_mode,
        dynamic_modules_path=dynamic_modules_path,
        trust_remote_code=trust_remote_code,
    )


# ===================================================================
# Module factory: script-based (local)
# ===================================================================


def _compat_local_script_module(
    path,
    download_mode=None,
    dynamic_modules_path=None,
    trust_remote_code=None,
) -> DatasetModule:
    """Load a dataset module from a local .py script (compat for datasets>=4.0)."""
    local_path = path
    name = Path(path).stem

    local_imports: List[Tuple[str, str]] = []
    imports = get_imports(local_path)
    for import_type, import_name, import_path, sub_directory in imports:
        if import_type == 'library':
            continue
        if import_type == 'internal':
            rel_path = os.path.join(
                os.path.dirname(local_path), import_path + '.py')
            if os.path.isfile(rel_path):
                local_imports.append((import_name, rel_path))
            elif os.path.isdir(
                    os.path.join(os.path.dirname(local_path), import_path)):
                local_imports.append(
                    (import_name,
                     os.path.join(os.path.dirname(local_path), import_path)))
        elif import_type == 'external':
            dl_config = DownloadConfig()
            dl_config.download_desc = 'Downloading extra modules'
            local_import_path = cached_path(
                import_path, download_config=dl_config)
            if sub_directory is not None:
                local_import_path = os.path.join(local_import_path,
                                                 sub_directory)
            local_imports.append((import_name, local_import_path))

    dynamic_modules_path = dynamic_modules_path or init_dynamic_modules()
    hash_val = files_to_hash([local_path] + [loc[1] for loc in local_imports])
    importable_file_path = _get_importable_file_path(
        dynamic_modules_path=dynamic_modules_path,
        module_namespace='datasets',
        subdirectory_name=hash_val,
        name=name,
    )
    if not os.path.exists(importable_file_path):
        trust = resolve_trust_remote_code(trust_remote_code, name)
        if trust:
            _create_importable_file(
                local_path=local_path,
                local_imports=local_imports,
                additional_files=[],
                dynamic_modules_path=dynamic_modules_path,
                module_namespace='datasets',
                subdirectory_name=hash_val,
                name=name,
                download_mode=download_mode,
            )
        else:
            raise ValueError(
                f'Loading {name} requires executing the dataset script. '
                'Set `trust_remote_code=True` to allow this.')
    module_path, hash_val = _load_importable_file(
        dynamic_modules_path=dynamic_modules_path,
        module_namespace='datasets',
        subdirectory_name=hash_val,
        name=name,
    )
    importlib.invalidate_caches()
    builder_kwargs = {
        'base_path': str(Path(path).resolve().parent),
    }
    return DatasetModule(module_path, hash_val, builder_kwargs)


# ===================================================================
# Module factory: without script (Hub)
# ===================================================================


def get_module_without_script(self) -> DatasetModule:
    """Monkey-patch target for ``HubDatasetModuleFactoryWithoutScript.get_module``."""
    revision = self.download_config.storage_options.get(
        'revision', None) or DEFAULT_DATASET_REVISION
    base_path = f"hf://datasets/{self.name}@{revision}/{self.data_dir or ''}".rstrip(
        '/')

    repo_id: str = self.name
    download_config = self.download_config.copy()

    dataset_readme_path = _download_repo_file(
        repo_id=repo_id,
        path_in_repo='README.md',
        download_config=download_config,
        revision=revision,
    )

    dataset_card_data = DatasetCard.load(
        Path(dataset_readme_path
             )).data if dataset_readme_path else DatasetCardData()
    subset_name: str = download_config.storage_options.get('name', None)

    metadata_configs = MetadataConfigs.from_dataset_card_data(
        dataset_card_data)
    dataset_infos = DatasetInfosDict.from_dataset_card_data(dataset_card_data)

    if self.data_files is not None:
        patterns = sanitize_patterns(self.data_files)
    elif metadata_configs and 'data_files' in next(
            iter(metadata_configs.values())):
        if subset_name is not None:
            subset_data_files = metadata_configs[subset_name]['data_files']
        else:
            subset_data_files = next(iter(
                metadata_configs.values()))['data_files']
        patterns = sanitize_patterns(subset_data_files)
    else:
        patterns = _get_data_patterns(
            base_path, download_config=self.download_config)

    data_files_dict = DataFilesDict.from_patterns(
        patterns,
        base_path=base_path,
        allowed_extensions=ALL_ALLOWED_EXTENSIONS,
        download_config=self.download_config,
    )
    module_name, default_builder_kwargs = infer_module_for_data_files(
        data_files=data_files_dict,
        path=self.name,
        download_config=self.download_config,
    )

    if hasattr(data_files_dict, 'filter'):
        data_files_dict = data_files_dict.filter(
            extensions=_MODULE_TO_EXTENSIONS[module_name])
    else:
        data_files_dict = data_files_dict.filter_extensions(
            _MODULE_TO_EXTENSIONS[module_name])

    module_path, _ = _PACKAGED_DATASETS_MODULES[module_name]

    if metadata_configs:
        supports_metadata = module_name in {'imagefolder', 'audiofolder'}
        create_builder_signature = inspect.signature(
            create_builder_configs_from_metadata_configs)
        in_args = {
            'module_path': module_path,
            'metadata_configs': metadata_configs,
            'base_path': base_path,
            'default_builder_kwargs': default_builder_kwargs,
            'download_config': self.download_config,
        }
        if 'supports_metadata' in create_builder_signature.parameters:
            in_args['supports_metadata'] = supports_metadata
        builder_configs, default_config_name = create_builder_configs_from_metadata_configs(
            **in_args)
    else:
        builder_configs: List[BuilderConfig] = [
            import_main_class(module_path).BUILDER_CONFIG_CLASS(
                data_files=data_files_dict,
                **default_builder_kwargs,
            )
        ]
        default_config_name = None

    api = _get_hub_api()
    endpoint = api.get_endpoint_for_read(
        repo_id=repo_id, repo_type=REPO_TYPE_DATASET)

    builder_kwargs = {
        'base_path':
        api.get_file_base_path(repo_id=repo_id, endpoint=endpoint),
        'repo_id': self.name,
        'dataset_name': camelcase_to_snakecase(Path(self.name).name),
        'data_files': data_files_dict,
    }
    download_config = self.download_config.copy()
    if download_config.download_desc is None:
        download_config.download_desc = 'Downloading metadata'

    if default_config_name is None and len(dataset_infos) == 1:
        default_config_name = next(iter(dataset_infos))

    return DatasetModule(
        module_path,
        revision,
        builder_kwargs,
        dataset_infos=dataset_infos,
        builder_configs_parameters=BuilderConfigsParameters(
            metadata_configs=metadata_configs,
            builder_configs=builder_configs,
            default_config_name=default_config_name,
        ),
    )
