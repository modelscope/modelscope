# noqa: isort:skip_file, yapf: disable
# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
"""ModelScope dataset loading orchestration.

This module provides :class:`DatasetsWrapperHF` and the
:func:`load_dataset_with_ctx` context manager that monkey-patch the
HuggingFace ``datasets`` library to work with the ModelScope Hub.

Sub-modules:
    _compat           – backward-compat shims for datasets>=4.0 script loading
    _module_factories – dataset module factory functions & data-file resolution
"""
import contextlib
import os
import warnings
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

from urllib.parse import urlencode

import requests
from datasets import (Dataset, DatasetBuilder, DatasetDict,
                      DownloadConfig, DownloadManager, DownloadMode, Features,
                      IterableDataset, IterableDatasetDict, Split,
                      VerificationMode, Version, config, data_files, LargeList,
                      Sequence as SequenceHf)

try:
    from datasets import List as DatasetList
except ImportError:
    DatasetList = None

from datasets.features import features
from datasets.features.features import _FEATURE_TYPES
from datasets.data_files import DataFilesDict, EmptyDatasetError
from datasets.exceptions import DataFilesNotFoundError, DatasetNotFoundError
from datasets.load import (
    CachedDatasetModuleFactory, DatasetModule,
    HubDatasetModuleFactoryWithParquetExport,
    PackagedDatasetModuleFactory,
    get_dataset_builder_class)
from datasets.packaged_modules import (_EXTENSION_TO_MODULE, _PACKAGED_DATASETS_MODULES)
from datasets.utils import file_utils
from datasets.utils.file_utils import (_raise_if_offline_mode_is_enabled,
                                       cached_path, relative_to_absolute_path)
from datasets.utils.info_utils import is_small_dataset
from datasets.utils.track import tracked_str

from huggingface_hub import hf_hub_url
from huggingface_hub.errors import OfflineModeIsEnabled
from huggingface_hub.hf_api import DatasetInfo as HfDatasetInfo
from huggingface_hub.hf_api import HfApi, RepoFile, RepoFolder
from huggingface_hub.hf_file_system import HfFileSystem

from modelscope import HubApi
from modelscope.hub.utils.utils import get_endpoint
from modelscope.msdatasets.utils.hf_file_utils import get_from_cache_ms
from modelscope.utils.config_ds import MS_DATASETS_CACHE
from modelscope.utils.constant import DEFAULT_DATASET_REVISION, REPO_TYPE_DATASET
from modelscope.utils.import_utils import has_attr_in_class
from modelscope.utils.file_utils import is_relative_path
from modelscope.utils.logger import get_logger

# -- Compat layer -----------------------------------------------------------
from modelscope.msdatasets.utils._compat import (
    _HAS_SCRIPT_LOADING,
    HubDatasetModuleFactoryWithScript,
    LocalDatasetModuleFactoryWithScript,
)

# -- Module factories --------------------------------------------------------
from modelscope.msdatasets.utils._module_factories import (
    _resolve_pattern,
    _download_repo_file,
    get_module_without_script,
    get_module_with_script,
    _compat_local_script_module,
    _compat_hub_script_module,
    _get_hub_api,
)

# Compatible with datasets 4.0+ (class name changed)
try:
    from datasets.load import (
        HubDatasetModuleFactory as HubDatasetModuleFactoryWithoutScript,
        LocalDatasetModuleFactory as LocalDatasetModuleFactoryWithoutScript)
except ImportError:
    from datasets.load import (
        HubDatasetModuleFactoryWithoutScript,
        LocalDatasetModuleFactoryWithoutScript)

logger = get_logger()


# ===================================================================
# Type definitions
# ===================================================================

ExpandDatasetProperty_T = Literal[
    'author',
    'cardData',
    'citation',
    'createdAt',
    'disabled',
    'description',
    'downloads',
    'downloadsAllTime',
    'gated',
    'lastModified',
    'likes',
    'paperswithcode_id',
    'private',
    'siblings',
    'sha',
    'tags',
]


# ===================================================================
# Feature patching (generate_from_dict_ms)
# ===================================================================

_NativeList = DatasetList if DatasetList is not None else SequenceHf


def generate_from_dict_ms(obj: Any):
    """Regenerate the nested feature object from a deserialized dict.

    This is a ModelScope-patched version of ``features.generate_from_dict``
    that handles backward compatibility for legacy ``Sequence`` types in
    datasets 4.0+ where ``Sequence`` is no longer a registered feature type.
    """
    if isinstance(obj, list):
        return [generate_from_dict_ms(value) for value in obj]
    if '_type' not in obj or isinstance(obj['_type'], dict):
        return {key: generate_from_dict_ms(value) for key, value in obj.items()}
    obj = dict(obj)
    _type = obj.pop('_type')

    if _type == 'Sequence':
        feature = obj.pop('feature')
        length = obj.get('length', -1)
        return SequenceHf(feature=generate_from_dict_ms(feature), length=length)

    class_type = _FEATURE_TYPES.get(_type, None) or globals().get(_type, None)

    if class_type is None:
        raise ValueError(f"Feature type '{_type}' not found. Available feature types: {list(_FEATURE_TYPES.keys())}")

    if class_type == LargeList:
        feature = obj.pop('feature')
        return LargeList(generate_from_dict_ms(feature), **obj)
    if _NativeList is not None and (class_type is _NativeList or issubclass(class_type, _NativeList)):
        feature = obj.pop('feature')
        return _NativeList(generate_from_dict_ms(feature), **obj)

    field_names = {f.name for f in fields(class_type)}
    return class_type(**{k: v for k, v in obj.items() if k in field_names})


# ===================================================================
# Download monkey-patch (_download_ms)
# ===================================================================

def _download_ms(self, url_or_filename: str, download_config: DownloadConfig) -> str:
    """ModelScope replacement for ``DownloadManager._download``.

    Rewrites relative paths and ``hf://`` URLs to ModelScope API endpoints.
    """
    url_or_filename = str(url_or_filename)
    if url_or_filename.startswith('hf://'):
        hf_path = url_or_filename[len('hf://'):]
        for _prefix in ('datasets/', 'models/'):
            if hf_path.startswith(_prefix):
                hf_path = hf_path[len(_prefix):]
                break
        if '@' in hf_path:
            at_idx = hf_path.index('@')
            after_at = hf_path[at_idx + 1:]
            slash_idx = after_at.find('/')
            if slash_idx == -1:
                revision = after_at
                file_path = ''
            else:
                revision = after_at[:slash_idx]
                file_path = after_at[slash_idx + 1:]
        else:
            parts = hf_path.split('/', 2)
            revision = DEFAULT_DATASET_REVISION
            file_path = parts[2] if len(parts) > 2 else ''
        params_str = urlencode({'Source': 'SDK', 'Revision': revision, 'FilePath': file_path})
        url_or_filename = self._base_path + params_str
    elif is_relative_path(url_or_filename):
        revision = DEFAULT_DATASET_REVISION
        params_str = urlencode({'Source': 'SDK', 'Revision': revision, 'FilePath': url_or_filename})
        url_or_filename = self._base_path + params_str

    out = cached_path(url_or_filename, download_config=download_config)
    out = tracked_str(out)
    out.set_origin(url_or_filename)
    return out


# ===================================================================
# HfApi monkey-patches (dataset_info, list_repo_tree, get_paths_info)
# ===================================================================

def _dataset_info(
    self,
    repo_id: str,
    *,
    revision: Optional[str] = None,
    timeout: Optional[float] = None,
    files_metadata: bool = False,
    token: Optional[Union[bool, str]] = None,
    expand: Optional[List[ExpandDatasetProperty_T]] = None,
) -> HfDatasetInfo:
    """ModelScope replacement for ``HfApi.dataset_info``."""
    repo_info_iter = self.list_repo_tree(
        repo_id=repo_id,
        path_in_repo='/',
        revision=revision,
        recursive=False,
        expand=expand,
        token=token,
        repo_type=REPO_TYPE_DATASET,
    )

    data_info = {
        'id': repo_id,
        'private': False,
        'author': repo_id.split('/')[0] if repo_id else None,
        'sha': revision,
        'lastModified': None,
        'gated': False,
        'disabled': False,
        'downloads': 0,
        'likes': 0,
        'tags': [],
        'cardData': [],
        'createdAt': None,
    }

    data_siblings = []
    for info_item in repo_info_iter:
        if isinstance(info_item, RepoFile):
            data_siblings.append(
                dict(
                    rfilename=info_item.rfilename,
                    blobId=info_item.blob_id,
                    size=info_item.size,
                )
            )
    data_info['siblings'] = data_siblings

    return HfDatasetInfo(**data_info)


# -- Repo tree cache ---------------------------------------------------------

_repo_tree_cache: Dict[tuple, List[Union[RepoFile, RepoFolder]]] = {}


def _derive_from_recursive_cache(
    repo_id: str,
    revision: str,
    path_in_repo: str,
    recursive: bool,
) -> Optional[List[Union[RepoFile, RepoFolder]]]:
    """Try to derive results from a cached recursive root listing."""
    root_key = (repo_id, revision, '/', True)
    root_cached = _repo_tree_cache.get(root_key)
    if root_cached is None:
        return None

    prefix = path_in_repo.strip('/') if path_in_repo and path_in_repo != '/' else ''
    results = []
    for item in root_cached:
        item_path = item.path
        if prefix:
            if not item_path.startswith(prefix + '/') and item_path != prefix:
                continue
            rel_path = item_path[len(prefix) + 1:] if item_path.startswith(prefix + '/') else ''
        else:
            rel_path = item_path
        if not recursive and '/' in rel_path:
            continue
        results.append(item)
    return results


def _list_repo_tree(
    self,
    repo_id: str,
    path_in_repo: Optional[str] = None,
    *,
    recursive: bool = True,
    expand: bool = False,
    revision: Optional[str] = None,
    repo_type: Optional[str] = None,
    token: Optional[Union[bool, str]] = None,
) -> Iterable[Union[RepoFile, RepoFolder]]:
    """ModelScope replacement for ``HfApi.list_repo_tree``."""
    revision = revision or DEFAULT_DATASET_REVISION
    normalized_path = path_in_repo or '/'
    cache_key = (repo_id, revision, normalized_path, recursive)

    cached = _repo_tree_cache.get(cache_key)
    if cached is not None:
        yield from cached
        return

    derived = _derive_from_recursive_cache(repo_id, revision, normalized_path, recursive)
    if derived is not None:
        _repo_tree_cache[cache_key] = derived
        yield from derived
        return

    api = _get_hub_api()
    endpoint = api.get_endpoint_for_read(
        repo_id=repo_id, repo_type=REPO_TYPE_DATASET)

    _owner, _dataset_name = repo_id.split('/')
    dataset_hub_id, _ = api.get_dataset_id_and_type(
        dataset_name=_dataset_name, namespace=_owner, endpoint=endpoint)

    results: List[Union[RepoFile, RepoFolder]] = []
    page_number = 1
    page_size = 500
    max_pages = 10000
    while page_number <= max_pages:
        try:
            dataset_files = api.get_dataset_files(
                repo_id=repo_id,
                revision=revision,
                root_path=normalized_path,
                recursive=recursive,
                page_number=page_number,
                page_size=page_size,
                endpoint=endpoint,
                dataset_hub_id=dataset_hub_id,
            )
        except Exception as e:
            logger.error(f'Get dataset: {repo_id} file list failed, message: {e}')
            break

        if not dataset_files:
            break

        for file_info_d in dataset_files:
            path_info = {
                'type': 'directory' if file_info_d['Type'] == 'tree' else 'file',
                'path': file_info_d['Path'],
                'size': file_info_d['Size'],
                'oid': file_info_d['Sha256'],
            }
            item = RepoFile(**path_info) if path_info['type'] == 'file' else RepoFolder(**path_info)
            results.append(item)
            yield item

        if len(dataset_files) < page_size:
            break
        page_number += 1

    _repo_tree_cache[cache_key] = results


def _get_paths_info(
    self,
    repo_id: str,
    paths: Union[List[str], str],
    *,
    expand: bool = False,
    revision: Optional[str] = None,
    repo_type: Optional[str] = None,
    token: Optional[Union[bool, str]] = None,
) -> List[Union[RepoFile, RepoFolder]]:
    """ModelScope replacement for ``HfApi.get_paths_info``."""
    revision = revision or DEFAULT_DATASET_REVISION
    if isinstance(paths, str):
        paths = [paths]
    paths_set = set(paths)

    # Check all available caches for matching paths
    for cache_key, cached_items in _repo_tree_cache.items():
        if cache_key[0] != repo_id or cache_key[1] != revision:
            continue
        matched = [item for item in cached_items if item.path in paths_set]
        if matched:
            return matched
        # Recursive root cache is authoritative – if paths not found, they don't exist
        if cache_key == (repo_id, revision, '/', True):
            return []

    repo_info_iter = self.list_repo_tree(
        repo_id=repo_id,
        recursive=False,
        expand=expand,
        revision=revision,
        repo_type=repo_type,
        token=token,
    )

    return [item for item in repo_info_iter if item.path in paths_set]


# ===================================================================
# HfFileSystem patch (_hf_fs_open)
# ===================================================================

_hf_fs_open_original = None


def _hf_fs_open(self, path, mode='rb', **kwargs):
    """Wrapper for HfFileSystem._open that fixes size=0 from ModelScope API.

    The ModelScope tree API may report Size=0 for files. When HfFileSystem
    caches this, AbstractBufferedFile treats the file as empty (0 bytes).
    This wrapper detects size=0 for files opened in read mode and resolves
    the actual size via a HEAD request before creating the file object.
    """
    if mode == 'rb' and 'size' not in kwargs:
        try:
            resolved = self.resolve_path(path)
            resolved_name = resolved.unresolve()
            parent = self._parent(resolved_name)
            cached_size = None
            if parent in self.dircache:
                for entry in self.dircache[parent]:
                    if entry['name'] == resolved_name and entry.get('type') == 'file':
                        cached_size = entry.get('size', -1)
                        break
            if cached_size == 0:
                url = hf_hub_url(
                    repo_id=resolved.repo_id,
                    revision=resolved.revision,
                    filename=resolved.path_in_repo,
                    repo_type=resolved.repo_type,
                    endpoint=self.endpoint,
                )
                headers = self._api._build_hf_headers()
                resp = requests.head(url, headers=headers, allow_redirects=True, timeout=30)
                if resp.status_code == 200:
                    cl = resp.headers.get('Content-Length')
                    if cl:
                        actual_size = int(cl)
                        kwargs['size'] = actual_size
                        for entry in self.dircache.get(parent, []):
                            if entry['name'] == resolved_name:
                                entry['size'] = actual_size
                                break
        except Exception:
            pass
    return _hf_fs_open_original(self, path, mode=mode, **kwargs)


# ===================================================================
# DatasetsWrapperHF
# ===================================================================

class DatasetsWrapperHF:

    @staticmethod
    def load_dataset(
        path: str,
        name: Optional[str] = None,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, Sequence[str],
                                   Mapping[str, Union[str,
                                                      Sequence[str]]]]] = None,
        split: Optional[Union[str, Split]] = None,
        cache_dir: Optional[str] = None,
        features: Optional[Features] = None,
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
        verification_mode: Optional[Union[VerificationMode, str]] = None,
        keep_in_memory: Optional[bool] = None,
        save_infos: bool = False,
        revision: Optional[Union[str, Version]] = None,
        token: Optional[Union[bool, str]] = None,
        use_auth_token='deprecated',
        task='deprecated',
        streaming: bool = False,
        num_proc: Optional[int] = None,
        storage_options: Optional[Dict] = None,
        trust_remote_code: bool = False,
        dataset_info_only: Optional[bool] = False,
        **config_kwargs,
    ) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset,
               dict]:

        if use_auth_token != 'deprecated':
            warnings.warn(
                "'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.\n"
                "You can remove this warning by passing 'token=<use_auth_token>' instead.",
                FutureWarning,
            )
            token = use_auth_token
        if task != 'deprecated':
            warnings.warn(
                "'task' was deprecated in version 2.13.0 and will be removed in 3.0.0.\n",
                FutureWarning,
            )
        else:
            task = None
        if data_files is not None and not data_files:
            raise ValueError(
                f"Empty 'data_files': '{data_files}'. It should be either non-empty or None (default)."
            )
        if Path(path, config.DATASET_STATE_JSON_FILENAME).exists():
            raise ValueError(
                'You are trying to load a dataset that was saved using `save_to_disk`. '
                'Please use `load_from_disk` instead.')

        if streaming and num_proc is not None:
            raise NotImplementedError(
                'Loading a streaming dataset in parallel with `num_proc` is not implemented. '
                'To parallelize streaming, you can wrap the dataset with a PyTorch DataLoader '
                'using `num_workers` > 1 instead.')

        download_mode = DownloadMode(download_mode
                                     or DownloadMode.REUSE_DATASET_IF_EXISTS)
        verification_mode = VerificationMode((
            verification_mode or VerificationMode.BASIC_CHECKS
        ) if not save_infos else VerificationMode.ALL_CHECKS)

        if trust_remote_code:
            logger.warning(f'Use trust_remote_code=True. Will invoke codes from {path}. Please make sure '
                           'that you can trust the external codes.')

        builder_instance = DatasetsWrapperHF.load_dataset_builder(
            path=path,
            name=name,
            data_dir=data_dir,
            data_files=data_files,
            cache_dir=cache_dir,
            features=features,
            download_config=download_config,
            download_mode=download_mode,
            revision=revision,
            token=token,
            storage_options=storage_options,
            trust_remote_code=trust_remote_code,
            _require_default_config_name=name is None,
            **config_kwargs,
        )

        if dataset_info_only:
            ret_dict = {}
            if isinstance(path, str) and path.endswith('.py') and os.path.exists(path):
                from datasets import get_dataset_config_names
                subset_list = get_dataset_config_names(path)
                ret_dict = {_subset: [] for _subset in subset_list}
                return ret_dict

            if builder_instance is None or not hasattr(builder_instance,
                                                       'builder_configs'):
                logger.error(f'No builder_configs found for {path} dataset.')
                return ret_dict

            _tmp_builder_configs = builder_instance.builder_configs
            for tmp_config_name, tmp_builder_config in _tmp_builder_configs.items():
                tmp_config_name = str(tmp_config_name)
                if hasattr(tmp_builder_config, 'data_files') and tmp_builder_config.data_files is not None:
                    ret_dict[tmp_config_name] = [str(item) for item in list(tmp_builder_config.data_files.keys())]
                else:
                    ret_dict[tmp_config_name] = []
            return ret_dict

        if streaming:
            return builder_instance.as_streaming_dataset(split=split)

        builder_instance.download_and_prepare(
            download_config=download_config,
            download_mode=download_mode,
            verification_mode=verification_mode,
            num_proc=num_proc,
            storage_options=storage_options,
        )

        keep_in_memory = (
            keep_in_memory if keep_in_memory is not None else is_small_dataset(
                builder_instance.info.dataset_size))
        ds = builder_instance.as_dataset(
            split=split,
            verification_mode=verification_mode,
            in_memory=keep_in_memory)
        if task is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                ds = ds.prepare_for_task(task)
        if save_infos:
            builder_instance._save_infos()

        try:
            api = _get_hub_api()
            if is_relative_path(path) and path.count('/') == 1:
                _namespace, _dataset_name = path.split('/')
                endpoint = api.get_endpoint_for_read(
                    repo_id=path, repo_type=REPO_TYPE_DATASET)
                api.dataset_download_statistics(dataset_name=_dataset_name, namespace=_namespace, endpoint=endpoint)
        except Exception as e:
            logger.warning(f'Could not record download statistics: {e}')

        return ds

    @staticmethod
    def load_dataset_builder(
        path: str,
        name: Optional[str] = None,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, Sequence[str],
                                   Mapping[str, Union[str,
                                                      Sequence[str]]]]] = None,
        cache_dir: Optional[str] = None,
        features: Optional[Features] = None,
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
        revision: Optional[Union[str, Version]] = None,
        token: Optional[Union[bool, str]] = None,
        use_auth_token='deprecated',
        storage_options: Optional[Dict] = None,
        trust_remote_code: Optional[bool] = None,
        _require_default_config_name=True,
        **config_kwargs,
    ) -> DatasetBuilder:

        if use_auth_token != 'deprecated':
            warnings.warn(
                "'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.\n"
                "You can remove this warning by passing 'token=<use_auth_token>' instead.",
                FutureWarning,
            )
            token = use_auth_token
        download_mode = DownloadMode(download_mode
                                     or DownloadMode.REUSE_DATASET_IF_EXISTS)
        if token is not None:
            download_config = download_config.copy(
            ) if download_config else DownloadConfig()
            download_config.token = token
        if storage_options is not None:
            download_config = download_config.copy(
            ) if download_config else DownloadConfig()
            download_config.storage_options.update(storage_options)

        dataset_module = DatasetsWrapperHF.dataset_module_factory(
            path,
            revision=revision,
            download_config=download_config,
            download_mode=download_mode,
            data_dir=data_dir,
            data_files=data_files,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            _require_default_config_name=_require_default_config_name,
            _require_custom_configs=bool(config_kwargs),
            name=name,
        )
        builder_kwargs = dataset_module.builder_kwargs
        data_dir = builder_kwargs.pop('data_dir', data_dir)
        data_files = builder_kwargs.pop('data_files', data_files)
        config_name = builder_kwargs.pop(
            'config_name', name
            or dataset_module.builder_configs_parameters.default_config_name)
        dataset_name = builder_kwargs.pop('dataset_name', None)
        info = dataset_module.dataset_infos.get(
            config_name) if dataset_module.dataset_infos else None

        if (path in _PACKAGED_DATASETS_MODULES and data_files is None
                and dataset_module.builder_configs_parameters.
                builder_configs[0].data_files is None):
            error_msg = f'Please specify the data files or data directory to load for the {path} dataset builder.'
            example_extensions = [
                extension for extension in _EXTENSION_TO_MODULE
                if _EXTENSION_TO_MODULE[extension] == path
            ]
            if example_extensions:
                error_msg += f'\nFor example `data_files={{"train": "path/to/data/train/*.{example_extensions[0]}"}}`'
            raise ValueError(error_msg)

        builder_cls = get_dataset_builder_class(
            dataset_module, dataset_name=dataset_name)

        builder_instance: DatasetBuilder = builder_cls(
            cache_dir=cache_dir,
            dataset_name=dataset_name,
            config_name=config_name,
            data_dir=data_dir,
            data_files=data_files,
            hash=dataset_module.hash,
            info=info,
            features=features,
            token=token,
            storage_options=storage_options,
            **builder_kwargs,
            **config_kwargs,
        )
        builder_instance._use_legacy_cache_dir_if_possible(dataset_module)

        return builder_instance

    @staticmethod
    def dataset_module_factory(
        path: str,
        revision: Optional[Union[str, Version]] = None,
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
        dynamic_modules_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[Dict, List, str, DataFilesDict]] = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: Optional[bool] = None,
        _require_default_config_name=True,
        _require_custom_configs=False,
        **download_kwargs,
    ) -> DatasetModule:

        subset_name: str = download_kwargs.pop('name', None)
        revision = revision or DEFAULT_DATASET_REVISION
        if download_config is None:
            download_config = DownloadConfig(**download_kwargs)
        download_config.storage_options.update({'name': subset_name})
        download_config.storage_options.update({'revision': revision})

        if download_config and download_config.cache_dir is None:
            download_config.cache_dir = MS_DATASETS_CACHE

        download_mode = DownloadMode(download_mode
                                     or DownloadMode.REUSE_DATASET_IF_EXISTS)
        download_config.extract_compressed_file = True
        download_config.force_extract = True
        download_config.force_download = download_mode == DownloadMode.FORCE_REDOWNLOAD

        filename = list(
            filter(lambda x: x,
                   path.replace(os.sep, '/').split('/')))[-1]
        if not filename.endswith('.py'):
            filename = filename + '.py'
        combined_path = os.path.join(path, filename)

        # Try packaged
        if path in _PACKAGED_DATASETS_MODULES:
            return PackagedDatasetModuleFactory(
                path,
                data_dir=data_dir,
                data_files=data_files,
                download_config=download_config,
                download_mode=download_mode,
            ).get_module()
        # Try locally with script
        elif path.endswith(filename):
            if os.path.isfile(path):
                if _HAS_SCRIPT_LOADING:
                    return LocalDatasetModuleFactoryWithScript(
                        path,
                        download_mode=download_mode,
                        dynamic_modules_path=dynamic_modules_path,
                        trust_remote_code=trust_remote_code,
                    ).get_module()
                return _compat_local_script_module(
                    path,
                    download_mode=download_mode,
                    dynamic_modules_path=dynamic_modules_path,
                    trust_remote_code=trust_remote_code,
                )
            else:
                raise FileNotFoundError(
                    f"Couldn't find a dataset script at {relative_to_absolute_path(path)}"
                )
        elif os.path.isfile(combined_path):
            if _HAS_SCRIPT_LOADING:
                return LocalDatasetModuleFactoryWithScript(
                    combined_path,
                    download_mode=download_mode,
                    dynamic_modules_path=dynamic_modules_path,
                    trust_remote_code=trust_remote_code,
                ).get_module()
            return _compat_local_script_module(
                combined_path,
                download_mode=download_mode,
                dynamic_modules_path=dynamic_modules_path,
                trust_remote_code=trust_remote_code,
            )
        elif os.path.isdir(path):
            return LocalDatasetModuleFactoryWithoutScript(
                path,
                data_dir=data_dir,
                data_files=data_files,
                download_mode=download_mode).get_module()
        # Try remotely
        elif is_relative_path(path) and path.count('/') == 1:
            try:
                _raise_if_offline_mode_is_enabled()

                try:
                    dataset_info = HfApi().dataset_info(
                        repo_id=path,
                        revision=revision,
                        token=download_config.token,
                        timeout=100.0,
                    )
                except Exception as e:  # noqa: broad exception from hf_hub
                    if isinstance(
                            e,
                        (  # noqa: E131
                            OfflineModeIsEnabled,
                            requests.exceptions.ConnectTimeout,
                            requests.exceptions.ConnectionError,
                        ),
                    ):
                        raise ConnectionError(
                            f"Couldn't reach '{path}' on the Hub ({type(e).__name__})"
                        )
                    elif '404' in str(e):
                        msg = f"Dataset '{path}' doesn't exist on the Hub"
                        raise DatasetNotFoundError(
                            msg
                            + f" at revision '{revision}'" if revision else msg
                        )
                    elif '401' in str(e):
                        msg = f"Dataset '{path}' doesn't exist on the Hub"
                        msg = msg + f" at revision '{revision}'" if revision else msg
                        raise DatasetNotFoundError(
                            msg + '. If the repo is private or gated, '
                            'make sure to log in with `huggingface-cli login`.'
                        )
                    else:
                        raise e

                dataset_readme_path = _download_repo_file(
                    repo_id=path,
                    path_in_repo='README.md',
                    download_config=download_config,
                    revision=revision,
                )
                commit_hash = os.path.basename(os.path.dirname(dataset_readme_path))

                if filename in [
                        sibling.rfilename for sibling in dataset_info.siblings
                ]:
                    can_load_config_from_parquet_export = False
                    if config.USE_PARQUET_EXPORT and can_load_config_from_parquet_export:
                        try:
                            if has_attr_in_class(HubDatasetModuleFactoryWithParquetExport, 'revision'):
                                return HubDatasetModuleFactoryWithParquetExport(
                                    path,
                                    revision=revision,
                                    download_config=download_config).get_module()

                            return HubDatasetModuleFactoryWithParquetExport(
                                path,
                                commit_hash=commit_hash,
                                download_config=download_config).get_module()
                        except Exception as e:
                            logger.error(e)

                    if _HAS_SCRIPT_LOADING:
                        if has_attr_in_class(HubDatasetModuleFactoryWithScript, 'revision'):
                            return HubDatasetModuleFactoryWithScript(
                                path,
                                revision=revision,
                                download_config=download_config,
                                download_mode=download_mode,
                                dynamic_modules_path=dynamic_modules_path,
                                trust_remote_code=trust_remote_code,
                            ).get_module()

                        return HubDatasetModuleFactoryWithScript(
                            path,
                            commit_hash=commit_hash,
                            download_config=download_config,
                            download_mode=download_mode,
                            dynamic_modules_path=dynamic_modules_path,
                            trust_remote_code=trust_remote_code,
                        ).get_module()

                    return _compat_hub_script_module(
                        path,
                        revision=revision,
                        download_config=download_config,
                        download_mode=download_mode,
                        dynamic_modules_path=dynamic_modules_path,
                        trust_remote_code=trust_remote_code,
                    )
                else:
                    if has_attr_in_class(HubDatasetModuleFactoryWithoutScript, 'revision'):
                        return HubDatasetModuleFactoryWithoutScript(
                            path,
                            revision=revision,
                            data_dir=data_dir,
                            data_files=data_files,
                            download_config=download_config,
                            download_mode=download_mode,
                        ).get_module()

                    return HubDatasetModuleFactoryWithoutScript(
                        path,
                        commit_hash=commit_hash,
                        data_dir=data_dir,
                        data_files=data_files,
                        download_config=download_config,
                        download_mode=download_mode,
                    ).get_module()
            except Exception as e1:
                logger.error(f'>> Error loading {path}: {e1}')

                try:
                    _cached_factory_kwargs = {'cache_dir': cache_dir}
                    if _HAS_SCRIPT_LOADING:
                        _cached_factory_kwargs['dynamic_modules_path'] = dynamic_modules_path
                    return CachedDatasetModuleFactory(
                        path, **_cached_factory_kwargs).get_module()
                except Exception:
                    if isinstance(e1, OfflineModeIsEnabled):
                        raise ConnectionError(
                            f"Couldn't reach the Hugging Face Hub for dataset '{path}': {e1}"
                        ) from None
                    if isinstance(e1,
                                  (DataFilesNotFoundError,
                                   DatasetNotFoundError, EmptyDatasetError)):
                        raise e1 from None
                    if isinstance(e1, FileNotFoundError):
                        raise FileNotFoundError(
                            f"Couldn't find a dataset script at {relative_to_absolute_path(combined_path)} or "
                            f'any data file in the same directory. '
                            f"Couldn't find '{path}' on the Hugging Face Hub either: {type(e1).__name__}: {e1}"
                        ) from None
                    raise e1 from None
        else:
            raise FileNotFoundError(
                f"Couldn't find a dataset script at {relative_to_absolute_path(combined_path)} or "
                f'any data file in the same directory.')


# ===================================================================
# Context manager – load_dataset_with_ctx
# ===================================================================

@contextlib.contextmanager
def load_dataset_with_ctx(*args, **kwargs):
    """Context manager that monkey-patches ``datasets`` to use ModelScope.

    All monkey-patches are applied on entry and restored on exit (for
    non-streaming mode) or kept alive (for streaming mode, where lazy
    iteration needs the patches to remain active).
    """
    global _hf_fs_open_original

    # Save originals
    hf_endpoint_origin = config.HF_ENDPOINT
    get_from_cache_origin = file_utils.get_from_cache
    _download_origin = DownloadManager._download if hasattr(DownloadManager, '_download') \
        else DownloadManager._download_single
    dataset_info_origin = HfApi.dataset_info
    list_repo_tree_origin = HfApi.list_repo_tree
    get_paths_info_origin = HfApi.get_paths_info
    resolve_pattern_origin = data_files.resolve_pattern
    get_module_without_script_origin = HubDatasetModuleFactoryWithoutScript.get_module
    get_module_with_script_origin = (
        HubDatasetModuleFactoryWithScript.get_module if _HAS_SCRIPT_LOADING else None)
    generate_from_dict_origin = features.generate_from_dict
    hf_fs_open_origin = HfFileSystem._open

    # Apply patches
    config.HF_ENDPOINT = get_endpoint()
    file_utils.get_from_cache = get_from_cache_ms
    if hasattr(DownloadManager, '_download'):
        DownloadManager._download = _download_ms
    else:
        DownloadManager._download_single = _download_ms
    HfApi.dataset_info = _dataset_info
    HfApi.list_repo_tree = _list_repo_tree
    HfApi.get_paths_info = _get_paths_info
    data_files.resolve_pattern = _resolve_pattern
    HubDatasetModuleFactoryWithoutScript.get_module = get_module_without_script
    if _HAS_SCRIPT_LOADING:
        HubDatasetModuleFactoryWithScript.get_module = get_module_with_script
    features.generate_from_dict = generate_from_dict_ms
    _hf_fs_open_original = hf_fs_open_origin
    HfFileSystem._open = _hf_fs_open

    streaming = kwargs.get('streaming', False)

    try:
        dataset_res = DatasetsWrapperHF.load_dataset(*args, **kwargs)
        yield dataset_res
    finally:
        _repo_tree_cache.clear()
        HubApi._dataset_id_type_cache.clear()

        HfFileSystem._open = hf_fs_open_origin
        _hf_fs_open_original = None

        if not streaming:
            config.HF_ENDPOINT = hf_endpoint_origin
            file_utils.get_from_cache = get_from_cache_origin
            features.generate_from_dict = generate_from_dict_origin

            if hasattr(DownloadManager, '_download'):
                DownloadManager._download = _download_origin
            else:
                DownloadManager._download_single = _download_origin

            HfApi.dataset_info = dataset_info_origin
            HfApi.list_repo_tree = list_repo_tree_origin
            HfApi.get_paths_info = get_paths_info_origin
            data_files.resolve_pattern = resolve_pattern_origin
            HubDatasetModuleFactoryWithoutScript.get_module = get_module_without_script_origin
            if _HAS_SCRIPT_LOADING:
                HubDatasetModuleFactoryWithScript.get_module = get_module_with_script_origin
