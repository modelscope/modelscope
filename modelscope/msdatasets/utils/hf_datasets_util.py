# noqa: isort:skip_file, yapf: disable
# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
"""ModelScope dataset loading with instance-isolated filesystems.

DatasetsWrapperHF reuses datasets builders and ModelScope Hub capabilities.
load_dataset_with_ctx preserves the context API without modifying global
Hugging Face state.
"""
import contextlib
import copy
import inspect
import os
import shutil
import warnings
from contextvars import ContextVar
from functools import partial
from pathlib import Path, PurePosixPath
from threading import RLock
from types import MethodType
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Union

import fsspec
import httpx
import requests
from datasets import (Dataset, DatasetBuilder, DatasetDict,
                      DownloadConfig, DownloadMode, Features,
                      IterableDataset, IterableDatasetDict, Split,
                      VerificationMode, Version, config, SplitDict)

from datasets.features import features
from datasets.data_files import DataFilesDict
from datasets.load import (
    CachedDatasetModuleFactory, DatasetModule,
    HubDatasetModuleFactoryWithParquetExport,
    PackagedDatasetModuleFactory,
    get_dataset_builder_class)
from datasets.packaged_modules import (_EXTENSION_TO_MODULE, _PACKAGED_DATASETS_MODULES)
from datasets.utils.file_utils import (_raise_if_offline_mode_is_enabled,
                                       relative_to_absolute_path)
from datasets.utils.info_utils import is_small_dataset

from huggingface_hub.errors import OfflineModeIsEnabled
from huggingface_hub.hf_api import DatasetInfo as HfDatasetInfo
from huggingface_hub.hf_api import HfApi, RepoFile, RepoFolder
from huggingface_hub.hf_file_system import HfFileSystem

from modelscope_hub.config import get_default_config

from modelscope import HubApi
from modelscope.hub.errors import NetworkError
from modelscope.msdatasets.utils.hf_file_utils import http_head_ms
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
    _download_repo_file,
    get_module_without_script,
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

def generate_from_dict_ms(obj: Any):
    """Preserve the legacy export using native Sequence and feature handling."""
    return features.generate_from_dict(obj)


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
    metadata = self._modelscope_api.get_repo(repo_id, REPO_TYPE_DATASET, revision=revision)
    return HfDatasetInfo(
        id=repo_id,
        private=bool(metadata.private or metadata.visibility == 'private'),
        author=metadata.owner or repo_id.split('/')[0],
        sha=revision or DEFAULT_DATASET_REVISION,
        gated=bool(metadata.gated),
        downloads=metadata.downloads,
        likes=metadata.likes,
        tags=metadata.tags,
        siblings=[dict(rfilename=item.path, blobId=item.blob_id, size=item.size)
                  for item in self.list_repo_tree(repo_id, revision=revision, recursive=False)
                  if isinstance(item, RepoFile)],
    )


# -- Repo tree cache ---------------------------------------------------------

_repo_tree_cache: Dict[tuple, List[Union[RepoFile, RepoFolder]]] = {}


def _derive_from_recursive_cache(
    repo_id: str,
    revision: str,
    path_in_repo: str,
    recursive: bool,
    cache: Optional[Dict] = None,
) -> Optional[List[Union[RepoFile, RepoFolder]]]:
    """Try to derive results from a cached recursive root listing."""
    root_key = (repo_id, revision, '/', True)
    root_cached = (cache if cache is not None else _repo_tree_cache).get(root_key)
    if root_cached is None:
        return None

    prefix = path_in_repo.strip('/') if path_in_repo and path_in_repo != '/' else ''
    results = []
    for item in root_cached:
        item_path = item.path
        if prefix:
            if not item_path.startswith(prefix + '/'):
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
    if repo_type not in (None, REPO_TYPE_DATASET):
        raise ValueError('ModelScope dataset filesystem only supports datasets')
    revision = revision or DEFAULT_DATASET_REVISION
    normalized_path = path_in_repo or '/'
    cache = self._modelscope_tree_cache
    cache_key = (repo_id, revision, normalized_path, recursive)
    with self._modelscope_tree_lock:
        results = cache.get(cache_key)
        if results is None:
            results = _derive_from_recursive_cache(
                repo_id, revision, normalized_path, recursive, cache=cache)
        if results is None:
            api = self._modelscope_api
            page_size = inspect.signature(api.get_dataset_files).parameters['page_size'].default
            page_number = 1
            entries = {}
            while True:
                page = api.get_dataset_files(
                    repo_id, revision=revision, root_path=normalized_path, recursive=recursive,
                    page_number=page_number, page_size=page_size, dataset_hub_id=repo_id)
                before = len(entries)
                for item in page:
                    entries[item['Path']] = item
                if not page or len(page) < page_size:
                    break
                if len(entries) == before:
                    raise RuntimeError('Dataset file listing pagination did not advance')
                page_number += 1
            results = [(RepoFolder if item['Type'] == 'tree' else RepoFile)(
                path=item['Path'], size=item.get('Size') or 0,
                oid=item.get('Sha256') or item.get('BlobId') or '') for item in entries.values()]
            cache[cache_key] = results
    yield from results


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
    paths_set = {paths} if isinstance(paths, str) else set(paths)
    parents = {str(PurePosixPath(path).parent) for path in paths_set}
    return [item for parent in sorted(parents) for item in self.list_repo_tree(
        repo_id, path_in_repo=parent if parent != '.' else '/', recursive=False,
        expand=expand, revision=revision, repo_type=repo_type, token=token)
        if item.path in paths_set]


# ===================================================================
# HfFileSystem patches (_hf_fs_open, _hf_fs_init)
# ===================================================================

class _ModelScopeHfApi(HfApi):
    """Adapt filesystem instances without modifying the native HfApi class."""

    def __init__(self, endpoint=None, token=None):
        super().__init__(endpoint=endpoint, token=False)
        self._modelscope_api: HubApi = HubApi(endpoint=endpoint, token=token)
        self._modelscope_token = token
        self._modelscope_tree_cache: Dict[tuple, List] = {}
        self._modelscope_tree_lock = RLock()

    dataset_info = _dataset_info
    list_repo_tree = _list_repo_tree
    get_paths_info = _get_paths_info

    def _build_hf_headers(self, token=None, **kwargs):
        headers = super()._build_hf_headers(token=False, **kwargs)
        cookies = self._modelscope_api.get_cookies(access_token=self._modelscope_token)
        if cookies and self._modelscope_token != '':
            headers['Cookie'] = '; '.join(f'{cookie.name}={cookie.value}' for cookie in cookies)
        return headers


def _modelscope_file_url(handle):
    resolved = handle.resolved_path
    namespace, name = resolved.repo_id.split('/', 1)
    return handle.fs._api._modelscope_api.get_dataset_file_url(
        resolved.path_in_repo, name, namespace, revision=resolved.revision,
        extension_filter=False, endpoint=handle.fs.endpoint)


_decoder_storage_options: ContextVar[Optional[Dict]] = ContextVar(
    'modelscope_dataset_decoder_options', default=None)


def _filesystem_options(options):
    config = get_default_config()
    defaults = {'endpoint': config.endpoint, 'token': config.token,
                'cache_dir': None, 'force_download': False,
                'local_files_only': False, 'repo_id': None}
    defaults.update(_decoder_storage_options.get() or {})
    defaults.update({key: value for key, value in options.items() if value is not None})
    return defaults


class _ModelScopeFileSystem(HfFileSystem):
    """Reuse native reads with instance-isolated endpoints, tokens and caches."""

    protocol = 'modelscope'

    @classmethod
    def _tokenize(cls, threading_ident, *args, **kwargs):
        return super()._tokenize(threading_ident, *args, **_filesystem_options(kwargs))

    def __init__(self, *args, endpoint=None, token=None, cache_dir=None,
                 force_download=None, local_files_only=None, repo_id=None, **kwargs):
        options = _filesystem_options({
            'endpoint': endpoint, 'token': token, 'cache_dir': cache_dir,
            'force_download': force_download, 'local_files_only': local_files_only,
            'repo_id': repo_id,
        })
        super().__init__(*args, endpoint=options['endpoint'], token=False, **kwargs)
        self._api = _ModelScopeHfApi(endpoint=options['endpoint'], token=options['token'])
        self.cache_dir = options['cache_dir']
        self.force_download = options['force_download']
        self.local_files_only = options['local_files_only']
        self._modelscope_repo_id: Optional[str] = options['repo_id']

    def _repo_and_revision_exist(self, repo_type, repo_id, revision):
        if (repo_type != REPO_TYPE_DATASET
                or (self._modelscope_repo_id is not None
                    and repo_id != self._modelscope_repo_id)):
            raise ValueError('Data files must belong to the requested ModelScope dataset')
        if self.local_files_only:
            return True, None
        return super()._repo_and_revision_exist(repo_type, repo_id, revision)

    def _open(self, path, mode='rb', **kwargs):
        if mode != 'rb':
            raise NotImplementedError('ModelScope dataset filesystem is read-only')
        _raise_if_offline_mode_is_enabled()
        resolved = self.resolve_path(path)
        if kwargs.get('block_size', self.block_size) != 0 and self.info(path).get('size') == 0:
            namespace, name = resolved.repo_id.split('/', 1)
            url = self._api._modelscope_api.get_dataset_file_url(
                resolved.path_in_repo, name, namespace, revision=resolved.revision,
                extension_filter=False, endpoint=self.endpoint)
            response = http_head_ms(url, headers=self._api._build_hf_headers())
            response.raise_for_status()
            kwargs['size'] = int(response.headers.get('Content-Length', 0))
        handle = super()._open(path, mode=mode, **kwargs)
        handle.url = MethodType(_modelscope_file_url, handle)
        return handle

    def get_file(self, rpath, lpath=None, callback=None, outfile=None, **kwargs):
        resolved = self.resolve_path(rpath, revision=kwargs.get('revision'))
        cached = self._api._modelscope_api.download_file(
            resolved.repo_id, REPO_TYPE_DATASET, resolved.path_in_repo,
            revision=resolved.revision, cache_dir=self.cache_dir,
            force=self.force_download, local_files_only=self.local_files_only)
        if outfile is None and hasattr(lpath, 'write'):
            outfile = lpath
        if outfile is not None:
            with open(cached, 'rb') as source:
                shutil.copyfileobj(source, outfile)
        elif lpath is not None:
            os.makedirs(os.path.dirname(os.fspath(lpath)) or '.', exist_ok=True)
            if os.path.abspath(cached) != os.path.abspath(lpath):
                shutil.copyfile(cached, lpath)
        else:
            raise ValueError('Either lpath or outfile must be provided')
        if callback is not None:
            callback.absolute_update(os.path.getsize(cached))

    def rm(self, *args, **kwargs):
        raise NotImplementedError('ModelScope dataset filesystem is read-only')

    def put_file(self, *args, **kwargs):
        raise NotImplementedError('ModelScope dataset filesystem is read-only')

    def cp_file(self, *args, **kwargs):
        raise NotImplementedError('ModelScope dataset filesystem is read-only')


fsspec.register_implementation('modelscope', _ModelScopeFileSystem)


def _decode_modelscope_feature(self, value, token_per_repo_id=None):
    """Keep decoder credentials on instances, outside serialized feature fields."""
    path = value.get('path') if isinstance(value, dict) else None
    if path and any(hop.startswith('modelscope://') for hop in path.split('::')):
        context_token = _decoder_storage_options.set(self._modelscope_storage_options)
        try:
            # Preserve lazy reads instead of preloading large media files.
            return type(self).decode_example(self, value, token_per_repo_id=token_per_repo_id)
        finally:
            _decoder_storage_options.reset(context_token)
    return type(self).decode_example(self, value, token_per_repo_id=token_per_repo_id)


def _bind_modelscope_decoders(dataset, storage_options):
    if 'modelscope' not in storage_options:
        return dataset
    options = copy.deepcopy(storage_options['modelscope'])

    def bind(feature):
        if isinstance(feature, dict):
            for child in feature.values():
                bind(child)
        elif isinstance(feature, (tuple, list)):
            for child in feature:
                bind(child)
        elif hasattr(feature, 'feature'):
            bind(feature.feature)
        if getattr(feature, 'decode', False) and hasattr(type(feature), 'decode_example'):
            feature._modelscope_storage_options = options
            feature.decode_example = partial(_decode_modelscope_feature, feature)

    for split in dataset.values() if isinstance(dataset, (DatasetDict, IterableDatasetDict)) else (dataset,):
        if split._info.features:
            bind(split._info.features)
    return dataset


class _DryRunDownloadManager:
    """Minimal download-manager stub for split discovery without I/O.

    Returns placeholder paths for all download/extract calls, allowing
    _split_generators() to execute and return SplitGenerator objects
    (which carry split names) without triggering actual network or disk I/O.
    """

    _PLACEHOLDER = os.devnull

    def download(self, url_or_urls):
        return self._map(url_or_urls)

    def download_and_extract(self, url_or_urls):
        return self._map(url_or_urls)

    def extract(self, path_or_paths):
        return self._map(path_or_paths)

    def download_custom(self, url_or_urls, custom_download):
        return self._map(url_or_urls)

    def iter_archive(self, path):
        return iter([])

    def iter_files(self, paths):
        return iter([])

    @property
    def manual_dir(self):
        return None

    @property
    def is_streaming(self):
        return False

    def _map(self, input_):
        if isinstance(input_, dict):
            return {k: self._PLACEHOLDER for k in input_}
        if isinstance(input_, (list, tuple, set)):
            return type(input_)(self._PLACEHOLDER for _ in input_)
        return self._PLACEHOLDER


def _discover_splits_from_builder(builder_instance):
    """Discover available split names by dry-running _split_generators().

    For script-based datasets that lack README split metadata, this calls
    the builder's _split_generators() with a stub download manager to
    extract split names without performing any actual downloads.

    Returns:
        A set of split name strings, or an empty set if discovery fails.
    """
    if not hasattr(builder_instance, '_split_generators'):
        return set()
    try:
        generators = builder_instance._split_generators(_DryRunDownloadManager())
        return {str(sg.name) for sg in generators}
    except Exception as e:
        logger.debug(f'Failed to discover splits from builder: {e}')
        return set()


def _validate_split_exists(builder_instance, split):
    """Fail-fast check: raise ValueError before downloading if the
    requested split does not exist in the dataset metadata.

    Args:
        builder_instance: The DatasetBuilder instance with info/config.
        split: The user-requested split specification (may be None).

    Raises:
        ValueError: If any requested split name is not found among
            the available splits declared in the dataset metadata.
    """
    if split is None:
        return

    from modelscope.msdatasets.utils._module_factories import _extract_split_names
    split_names = _extract_split_names(split)
    if not split_names:
        return

    # Source 1: info.splits (original metadata from README)
    available = set()
    info = getattr(builder_instance, 'info', None)
    if info is not None and info.splits:
        available = set(info.splits.keys())

    # Source 2: config.data_files keys
    if not available:
        config = getattr(builder_instance, 'config', None)
        data_files = getattr(config, 'data_files', None)
        if isinstance(data_files, dict):
            available = set(data_files.keys())

    # Source 3: dry-run _split_generators() for script-based datasets
    if not available:
        available = _discover_splits_from_builder(builder_instance)

    if not available:
        logger.debug(
            'Cannot determine available splits from dataset metadata; '
            'split validation skipped. Invalid splits will be caught downstream.'
        )
        return

    missing = split_names - available
    if missing:
        raise ValueError(
            f'Split {sorted(missing)} not found in dataset. '
            f'Available splits: {sorted(available)}'
        )


def _align_builder_splits_with_data_files(builder_instance, split):
    """Align builder.info.splits with the actually requested split(s).

    When data_files have been filtered to a subset of splits (see
    _filter_data_files_by_split in _module_factories.py), the builder's
    info.splits metadata may still list all original splits from the
    README.  download_and_prepare() calls verify_splits() which would
    then raise ExpectedMoreSplitsError.  This helper prunes info.splits
    to only contain the splits that will actually be generated.
    """
    if split is None:
        return
    info = getattr(builder_instance, 'info', None)
    if info is None or info.splits is None:
        return

    from modelscope.msdatasets.utils._module_factories import _extract_split_names
    split_names = _extract_split_names(split)
    if not split_names:
        return

    existing_keys = set(info.splits.keys())
    if split_names >= existing_keys:
        return  # All splits requested, no filtering needed

    filtered = {k: v for k, v in info.splits.items() if k in split_names}
    if not filtered:
        return  # Safety: don't empty out splits

    info.splits = SplitDict(filtered, dataset_name=info.splits.dataset_name)


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
            split=split,
            **config_kwargs,
        )

        if dataset_info_only:
            ret_dict = {}

            # Case 1: Local .py script file
            if isinstance(path, str) and path.endswith('.py') and os.path.exists(path):
                configs = builder_instance.builder_configs or {builder_instance.config.name: builder_instance.config}
                for config_name in configs:
                    candidate = DatasetsWrapperHF.load_dataset_builder(
                        path, name=config_name, data_dir=data_dir, data_files=data_files,
                        cache_dir=cache_dir, features=features, download_config=download_config,
                        download_mode=download_mode, revision=revision, token=token,
                        storage_options=storage_options, trust_remote_code=trust_remote_code,
                        **config_kwargs)
                    ret_dict[config_name] = sorted(_discover_splits_from_builder(candidate))
                return ret_dict

            if builder_instance is None:
                logger.error(f'No builder instance created for {path} dataset.')
                return ret_dict

            # Case 2: Try builder_configs with data_files (packaged datasets)
            _tmp_builder_configs = getattr(builder_instance, 'builder_configs', None)
            if _tmp_builder_configs:
                if hasattr(_tmp_builder_configs, 'items'):
                    configs_iter = _tmp_builder_configs.items()
                else:
                    configs_iter = [(getattr(c, 'name', 'default'), c) for c in _tmp_builder_configs]
                for tmp_config_name, tmp_builder_config in configs_iter:
                    tmp_config_name = str(tmp_config_name)
                    if hasattr(tmp_builder_config, 'data_files') and tmp_builder_config.data_files is not None:
                        ret_dict[tmp_config_name] = [str(item) for item in list(tmp_builder_config.data_files.keys())]

            # Case 3: Fallback for script datasets — use info.splits or dry-run discovery
            if not ret_dict or all(not v for v in ret_dict.values()):
                config_name = getattr(builder_instance, 'config_name', 'default') or 'default'
                splits = []

                # Try info.splits (from README metadata)
                info = getattr(builder_instance, 'info', None)
                if info is not None and info.splits:
                    splits = sorted(info.splits.keys())

                # Fallback: dry-run _split_generators()
                if not splits:
                    discovered = _discover_splits_from_builder(builder_instance)
                    if discovered:
                        splits = sorted(discovered)

                if splits:
                    ret_dict = {config_name: splits}
                elif not ret_dict:
                    ret_dict = {config_name: []}

            return ret_dict

        _validate_split_exists(builder_instance, split)

        if streaming:
            return _bind_modelscope_decoders(
                builder_instance.as_streaming_dataset(split=split), builder_instance.storage_options)

        _align_builder_splits_with_data_files(builder_instance, split)

        download_config = download_config.copy() if download_config else DownloadConfig()
        download_config.token = builder_instance.token
        download_config.storage_options = copy.deepcopy(builder_instance.storage_options or {})
        download_config.cache_dir = download_config.cache_dir or cache_dir or MS_DATASETS_CACHE
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
        as_dataset_kwargs = {'split': split, 'in_memory': keep_in_memory}
        if 'verification_mode' in inspect.signature(builder_instance.as_dataset).parameters:
            as_dataset_kwargs['verification_mode'] = verification_mode
        ds = builder_instance.as_dataset(**as_dataset_kwargs)
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

        return _bind_modelscope_decoders(ds, builder_instance.storage_options)

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
        split: Optional[Union[str, Split]] = None,
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
        download_config = download_config.copy() if download_config else DownloadConfig()
        download_config.storage_options = copy.deepcopy(download_config.storage_options or {})
        if token is not None:
            download_config.token = token
        if storage_options is not None:
            download_config.storage_options.update(copy.deepcopy(storage_options))

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
            split=split,
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
        builder_kwargs = {**builder_kwargs, **config_kwargs}
        ms_repo_id = None
        if str(builder_kwargs.get('base_path', '')).startswith('modelscope://'):
            # datasets 5.0.1 validates only hf:// paths; our filesystem enforces repo scope.
            ms_repo_id = builder_kwargs.pop('repo_id', None)

        builder_instance: DatasetBuilder = builder_cls(
            cache_dir=cache_dir,
            dataset_name=dataset_name,
            config_name=config_name,
            data_dir=data_dir,
            data_files=data_files,
            hash=dataset_module.hash,
            info=info,
            features=features,
            token=download_config.token,
            storage_options=download_config.storage_options,
            **builder_kwargs,
        )
        if ms_repo_id is not None:
            builder_instance.repo_id = ms_repo_id
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
        split: Optional[Union[str, Split]] = None,
        **download_kwargs,
    ) -> DatasetModule:

        subset_name: str = download_kwargs.pop('name', None)
        revision = revision or DEFAULT_DATASET_REVISION
        if download_config is None:
            download_config = DownloadConfig(**download_kwargs)
        if download_config.cache_dir is None:
            download_config.cache_dir = cache_dir or MS_DATASETS_CACHE

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
                    download_config=download_config,
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
                download_config=download_config,
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

                options = dict(download_config.storage_options.get('modelscope', {}))
                ms_token = options.get('token',
                                       download_config.token if isinstance(download_config.token, str) else None)
                ms_token = get_default_config().token if ms_token is None else ms_token
                api = HubApi(endpoint=options.get('endpoint'), token=ms_token)
                endpoint = api.get_endpoint_for_read(path, repo_type=REPO_TYPE_DATASET, token=ms_token)
                download_config.storage_options['modelscope'] = {
                    **options, 'endpoint': endpoint, 'token': ms_token, 'repo_id': path,
                    'cache_dir': download_config.cache_dir,
                    'force_download': download_config.force_download,
                    'local_files_only': getattr(download_config, 'local_files_only', False),
                }
                download_config.token = False
                dataset_info = _ModelScopeHfApi(endpoint=endpoint, token=ms_token).dataset_info(
                    path, revision=revision)

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
                    factory_kwargs = {'revision': revision} if has_attr_in_class(
                        HubDatasetModuleFactoryWithoutScript, 'revision') else {'commit_hash': commit_hash}
                    return get_module_without_script(HubDatasetModuleFactoryWithoutScript(
                        path, data_dir=data_dir, data_files=data_files,
                        download_config=download_config, download_mode=download_mode,
                        **factory_kwargs), name=subset_name, revision=revision, split=split)
            except (OfflineModeIsEnabled, NetworkError, requests.exceptions.RequestException,
                    httpx.TransportError) as error:
                try:
                    cached = CachedDatasetModuleFactory(path, cache_dir=cache_dir).get_module()
                except (FileNotFoundError, ValueError):
                    if isinstance(error, (OfflineModeIsEnabled, httpx.TransportError,
                                          requests.exceptions.RequestException)):
                        raise ConnectionError(f"Couldn't reach ModelScope dataset '{path}'") from error
                    raise error
                if not cached.module_path.startswith('datasets.packaged_modules.'):
                    from modelscope.msdatasets.utils._compat import resolve_trust_remote_code
                    resolve_trust_remote_code(trust_remote_code, path)
                return cached
        else:
            raise FileNotFoundError(
                f"Couldn't find a dataset script at {relative_to_absolute_path(combined_path)} or "
                f'any data file in the same directory.')


# ===================================================================
# Context manager – load_dataset_with_ctx
# ===================================================================

@contextlib.contextmanager
def load_dataset_with_ctx(*args, **kwargs):
    """Preserve the context API with filesystem-local state and no global patches."""
    yield DatasetsWrapperHF.load_dataset(*args, **kwargs)
