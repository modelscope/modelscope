# noqa: isort:skip_file, yapf: disable
# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
import importlib
import contextlib
import os
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union, Tuple

from urllib.parse import urlencode

import requests
from datasets import (BuilderConfig, Dataset, DatasetBuilder, DatasetDict,
                      DownloadConfig, DownloadManager, DownloadMode, Features,
                      IterableDataset, IterableDatasetDict, Split,
                      VerificationMode, Version, config, data_files)
from datasets.data_files import (
    FILES_TO_IGNORE, DataFilesDict, DataFilesList, EmptyDatasetError,
    _get_data_files_patterns, _is_inside_unrequested_special_dir,
    _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir, get_metadata_patterns, sanitize_patterns)
from datasets.download.streaming_download_manager import (
    _prepare_path_and_storage_options, xbasename, xjoin)
from datasets.exceptions import DataFilesNotFoundError, DatasetNotFoundError
from datasets.info import DatasetInfosDict
from datasets.load import (
    ALL_ALLOWED_EXTENSIONS, BuilderConfigsParameters,
    CachedDatasetModuleFactory, DatasetModule,
    HubDatasetModuleFactoryWithoutScript,
    HubDatasetModuleFactoryWithParquetExport,
    HubDatasetModuleFactoryWithScript, LocalDatasetModuleFactoryWithoutScript,
    LocalDatasetModuleFactoryWithScript, PackagedDatasetModuleFactory,
    create_builder_configs_from_metadata_configs, get_dataset_builder_class,
    import_main_class, infer_module_for_data_files, files_to_hash,
    _get_importable_file_path, resolve_trust_remote_code, _create_importable_file, _load_importable_file,
    init_dynamic_modules)
from datasets.naming import camelcase_to_snakecase
from datasets.packaged_modules import (_EXTENSION_TO_MODULE,
                                       _MODULE_SUPPORTS_METADATA,
                                       _MODULE_TO_EXTENSIONS,
                                       _PACKAGED_DATASETS_MODULES)
from datasets.utils import _datasets_server, file_utils
from datasets.utils.file_utils import (OfflineModeIsEnabled,
                                       _raise_if_offline_mode_is_enabled,
                                       cached_path, is_local_path,
                                       is_relative_path,
                                       relative_to_absolute_path)
from datasets.utils.info_utils import is_small_dataset
from datasets.utils.metadata import MetadataConfigs
from datasets.utils.py_utils import get_imports
from datasets.utils.track import tracked_str
from fsspec import filesystem
from fsspec.core import _un_chain
from fsspec.utils import stringify_path
from huggingface_hub import (DatasetCard, DatasetCardData)
from huggingface_hub.hf_api import DatasetInfo as HfDatasetInfo
from huggingface_hub.hf_api import HfApi, RepoFile, RepoFolder
from packaging import version

from modelscope import HubApi
from modelscope.hub.utils.utils import get_endpoint
from modelscope.msdatasets.utils.hf_file_utils import get_from_cache_ms
from modelscope.utils.config_ds import MS_DATASETS_CACHE
from modelscope.utils.constant import DEFAULT_DATASET_NAMESPACE
from modelscope.utils.logger import get_logger

logger = get_logger()


def _download_ms(self, url_or_filename: str, download_config: DownloadConfig) -> str:
    url_or_filename = str(url_or_filename)
    # for temp val
    revision = None
    if url_or_filename.startswith('hf://'):
        revision, url_or_filename = url_or_filename.split('@', 1)[-1].split('/', 1)
    if is_relative_path(url_or_filename):
        # append the relative path to the base_path
        # url_or_filename = url_or_path_join(self._base_path, url_or_filename)
        revision = revision or 'master'
        # Note: make sure the FilePath is the last param
        params: dict = {'Source': 'SDK', 'Revision': revision, 'FilePath': url_or_filename}
        params: str = urlencode(params)
        url_or_filename = self._base_path + params

    out = cached_path(url_or_filename, download_config=download_config)
    out = tracked_str(out)
    out.set_origin(url_or_filename)
    return out


def _dataset_info(
    self,
    repo_id: str,
    *,
    revision: Optional[str] = None,
    timeout: Optional[float] = None,
    files_metadata: bool = False,
    token: Optional[Union[bool, str]] = None,
) -> HfDatasetInfo:
    """
    Get info on one specific dataset on huggingface.co.

    Dataset can be private if you pass an acceptable token.

    Args:
        repo_id (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        revision (`str`, *optional*):
            The revision of the dataset repository from which to get the
            information.
        timeout (`float`, *optional*):
            Whether to set a timeout for the request to the Hub.
        files_metadata (`bool`, *optional*):
            Whether or not to retrieve metadata for files in the repository
            (size, LFS metadata, etc). Defaults to `False`.
        token (`bool` or `str`, *optional*):
            A valid authentication token (see https://huggingface.co/settings/token).
            If `None` or `True` and machine is logged in (through `huggingface-cli login`
            or [`~huggingface_hub.login`]), token will be retrieved from the cache.
            If `False`, token is not sent in the request header.

    Returns:
        [`hf_api.DatasetInfo`]: The dataset repository information.

    <Tip>

    Raises the following errors:

        - [`~utils.RepositoryNotFoundError`]
          If the repository to download from cannot be found. This may be because it doesn't exist,
          or because it is set to `private` and you do not have access.
        - [`~utils.RevisionNotFoundError`]
          If the revision to download from cannot be found.

    </Tip>
    """
    _api = HubApi()
    _namespace, _dataset_name = repo_id.split('/')
    dataset_hub_id, dataset_type = _api.get_dataset_id_and_type(
        dataset_name=_dataset_name, namespace=_namespace)

    revision: str = revision or 'master'
    data = _api.get_dataset_infos(dataset_hub_id=dataset_hub_id,
                                  revision=revision,
                                  files_metadata=files_metadata,
                                  timeout=timeout)

    # Parse data
    data_d: dict = data['Data']
    data_file_list: list = data_d['Files']
    # commit_info: dict = data_d['LatestCommitter']

    # Update data   # TODO: columns align with HfDatasetInfo
    data['id'] = repo_id
    data['private'] = False
    data['author'] = repo_id.split('/')[0] if repo_id else None
    data['sha'] = revision
    data['lastModified'] = None
    data['gated'] = False
    data['disabled'] = False
    data['downloads'] = 0
    data['likes'] = 0
    data['tags'] = []
    data['cardData'] = []
    data['createdAt'] = None

    # e.g. {'rfilename': 'xxx', 'blobId': 'xxx', 'size': 0, 'lfs': {'size': 0, 'sha256': 'xxx', 'pointerSize': 0}}
    data['siblings'] = []
    for file_info_d in data_file_list:
        file_info = {
            'rfilename': file_info_d['Path'],
            'blobId': file_info_d['Id'],
            'size': file_info_d['Size'],
            'type': 'directory' if file_info_d['Type'] == 'tree' else 'file',
            'lfs': {
                'size': file_info_d['Size'],
                'sha256': file_info_d['Sha256'],
                'pointerSize': 0
            }
        }
        data['siblings'].append(file_info)

    return HfDatasetInfo(**data)


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

    _api = HubApi()

    if is_relative_path(repo_id) and repo_id.count('/') == 1:
        _namespace, _dataset_name = repo_id.split('/')
    elif is_relative_path(repo_id) and repo_id.count('/') == 0:
        logger.warning(f'Got a relative path: {repo_id} without namespace, '
                       f'Use default namespace: {DEFAULT_DATASET_NAMESPACE}')
        _namespace, _dataset_name = DEFAULT_DATASET_NAMESPACE, repo_id
    else:
        raise ValueError(f'Invalid repo_id: {repo_id} !')

    data: dict = _api.list_repo_tree(dataset_name=_dataset_name,
                                     namespace=_namespace,
                                     revision=revision or 'master',
                                     root_path=path_in_repo or None,
                                     recursive=True,
                                     )
    # Parse data
    # Type: 'tree' or 'blob'
    data_d: dict = data['Data']
    data_file_list: list = data_d['Files']
    # commit_info: dict = data_d['LatestCommitter']

    for file_info_d in data_file_list:
        path_info = {}
        path_info[
            'type'] = 'directory' if file_info_d['Type'] == 'tree' else 'file'
        path_info['path'] = file_info_d['Path']
        path_info['size'] = file_info_d['Size']
        path_info['oid'] = file_info_d['Sha256']

        yield RepoFile(
            **path_info) if path_info['type'] == 'file' else RepoFolder(
                **path_info)


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

    _api = HubApi()
    _namespace, _dataset_name = repo_id.split('/')
    dataset_hub_id, dataset_type = _api.get_dataset_id_and_type(
        dataset_name=_dataset_name, namespace=_namespace)

    revision: str = revision or 'master'
    data = _api.get_dataset_infos(dataset_hub_id=dataset_hub_id,
                                  revision=revision,
                                  files_metadata=False,
                                  recursive='False')
    data_d: dict = data['Data']
    data_file_list: list = data_d['Files']

    return [
        RepoFile(path=item_d['Name'],
                 size=item_d['Size'],
                 oid=item_d['Revision'],
                 lfs=None,           # TODO: lfs type to be supported
                 last_commit=None,   # TODO: lfs type to be supported
                 security=None
                 ) for item_d in data_file_list if item_d['Name'] == 'README.md'
    ]


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
    # Reverse iterate the chain, creating a nested target_* structure
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
    """
    Resolve the paths and URLs of the data files from the pattern passed by the user.

    You can use patterns to resolve multiple local files. Here are a few examples:
    - *.csv to match all the CSV files at the first level
    - **.csv to match all the CSV files at any level
    - data/* to match all the files inside "data"
    - data/** to match all the files inside "data" and its subdirectories

    The patterns are resolved using the fsspec glob.

    glob.glob, Path.glob, Path.match or fnmatch do not support ** with a prefix/suffix other than a forward slash /.
    For instance, this means **.json is the same as *.json. On the contrary, the fsspec glob has no limits regarding the ** prefix/suffix,  # noqa: E501
    resulting in **.json being equivalent to **/*.json.

    More generally:
    - '*' matches any character except a forward-slash (to match just the file or directory name)
    - '**' matches any character including a forward-slash /

    Hidden files and directories (i.e. whose names start with a dot) are ignored, unless they are explicitly requested.
    The same applies to special directories that start with a double underscore like "__pycache__".
    You can still include one if the pattern explicilty mentions it:
    - to include a hidden file: "*/.hidden.txt" or "*/.*"
    - to include a hidden directory: ".hidden/*" or ".*/*"
    - to include a special directory: "__special__/*" or "__*/*"

    Example::

        >>> from datasets.data_files import resolve_pattern
        >>> base_path = "."
        >>> resolve_pattern("docs/**/*.py", base_path)
        [/Users/mariosasko/Desktop/projects/datasets/docs/source/_config.py']

    Args:
        pattern (str): Unix pattern or paths or URLs of the data files to resolve.
            The paths can be absolute or relative to base_path.
            Remote filesystems using fsspec are supported, e.g. with the hf:// protocol.
        base_path (str): Base path to use when resolving relative paths.
        allowed_extensions (Optional[list], optional): White-list of file extensions to use. Defaults to None (all extensions).
            For example: allowed_extensions=[".csv", ".json", ".txt", ".parquet"]
    Returns:
        List[str]: List of paths or URLs to the local or remote files that match the patterns.
    """
    if is_relative_path(pattern):
        pattern = xjoin(base_path, pattern)
    elif is_local_path(pattern):
        base_path = os.path.splitdrive(pattern)[0] + os.sep
    else:
        base_path = ''
    # storage_options： {'hf': {'token': None, 'endpoint': 'https://huggingface.co'}}
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
        # 10 times faster glob with detail=True (ignores costly info like lastCommit)
        glob_kwargs['expand_info'] = False

    tmp_file_paths = fs.glob(pattern, detail=True, **glob_kwargs)

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
    ]  # ignore .ipynb and __pycache__, but keep /../
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
        download_config: Optional[DownloadConfig] = None) -> Dict[str,
                                                                  List[str]]:
    """
    Get the default pattern from a directory testing all the supported patterns.
    The first patterns to return a non-empty list of data files is returned.

    Some examples of supported patterns:

    Input:

        my_dataset_repository/
        ├── README.md
        └── dataset.csv

    Output:

        {"train": ["**"]}

    Input:

        my_dataset_repository/
        ├── README.md
        ├── train.csv
        └── test.csv

        my_dataset_repository/
        ├── README.md
        └── data/
            ├── train.csv
            └── test.csv

        my_dataset_repository/
        ├── README.md
        ├── train_0.csv
        ├── train_1.csv
        ├── train_2.csv
        ├── train_3.csv
        ├── test_0.csv
        └── test_1.csv

    Output:

        {'train': ['train[-._ 0-9/]**', '**/*[-._ 0-9/]train[-._ 0-9/]**',
                    'training[-._ 0-9/]**', '**/*[-._ 0-9/]training[-._ 0-9/]**'],
         'test': ['test[-._ 0-9/]**', '**/*[-._ 0-9/]test[-._ 0-9/]**',
                    'testing[-._ 0-9/]**', '**/*[-._ 0-9/]testing[-._ 0-9/]**', ...]}

    Input:

        my_dataset_repository/
        ├── README.md
        └── data/
            ├── train/
            │   ├── shard_0.csv
            │   ├── shard_1.csv
            │   ├── shard_2.csv
            │   └── shard_3.csv
            └── test/
                ├── shard_0.csv
                └── shard_1.csv

    Output:

        {'train': ['train[-._ 0-9/]**', '**/*[-._ 0-9/]train[-._ 0-9/]**',
                'training[-._ 0-9/]**', '**/*[-._ 0-9/]training[-._ 0-9/]**'],
         'test': ['test[-._ 0-9/]**', '**/*[-._ 0-9/]test[-._ 0-9/]**',
                'testing[-._ 0-9/]**', '**/*[-._ 0-9/]testing[-._ 0-9/]**', ...]}

    Input:

        my_dataset_repository/
        ├── README.md
        └── data/
            ├── train-00000-of-00003.csv
            ├── train-00001-of-00003.csv
            ├── train-00002-of-00003.csv
            ├── test-00000-of-00001.csv
            ├── random-00000-of-00003.csv
            ├── random-00001-of-00003.csv
            └── random-00002-of-00003.csv

    Output:

        {'train': ['data/train-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]*.*'],
         'test': ['data/test-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]*.*'],
         'random': ['data/random-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]*.*']}

    In order, it first tests if SPLIT_PATTERN_SHARDED works, otherwise it tests the patterns in ALL_DEFAULT_PATTERNS.
    """
    resolver = partial(
        _resolve_pattern, base_path=base_path, download_config=download_config)
    try:
        return _get_data_files_patterns(resolver)
    except FileNotFoundError:
        raise EmptyDatasetError(
            f"The directory at {base_path} doesn't contain any data files"
        ) from None


def get_module_without_script(self) -> DatasetModule:
    _ms_api = HubApi()
    _repo_id: str = self.name
    _namespace, _dataset_name = _repo_id.split('/')

    # hfh_dataset_info = HfApi(config.HF_ENDPOINT).dataset_info(
    #     self.name,
    #     revision=self.revision,
    #     token=self.download_config.token,
    #     timeout=100.0,
    # )
    # even if metadata_configs is not None (which means that we will resolve files for each config later)
    # we cannot skip resolving all files because we need to infer module name by files extensions
    # revision = hfh_dataset_info.sha  # fix the revision in case there are new commits in the meantime
    revision = self.revision or 'master'
    base_path = f"hf://datasets/{self.name}@{revision}/{self.data_dir or ''}".rstrip(
        '/')

    download_config = self.download_config.copy()
    if download_config.download_desc is None:
        download_config.download_desc = 'Downloading readme'
    try:
        url_or_filename = _ms_api.get_dataset_file_url(
            file_name='README.md',
            dataset_name=_dataset_name,
            namespace=_namespace,
            revision=revision,
            extension_filter=False,
        )

        dataset_readme_path = cached_path(
            url_or_filename=url_or_filename, download_config=download_config)
        dataset_card_data = DatasetCard.load(Path(dataset_readme_path)).data
    except FileNotFoundError:
        dataset_card_data = DatasetCardData()

    subset_name: str = download_config.storage_options.get('name', None)

    metadata_configs = MetadataConfigs.from_dataset_card_data(
        dataset_card_data)
    dataset_infos = DatasetInfosDict.from_dataset_card_data(dataset_card_data)
    # we need a set of data files to find which dataset builder to use
    # because we need to infer module name by files extensions
    if self.data_files is not None:
        patterns = sanitize_patterns(self.data_files)
    elif metadata_configs and 'data_files' in next(
            iter(metadata_configs.values())):

        if subset_name is not None:
            subset_data_files = metadata_configs[subset_name]['data_files']
        else:
            subset_data_files = next(iter(metadata_configs.values()))['data_files']
        patterns = sanitize_patterns(subset_data_files)
    else:
        patterns = _get_data_patterns(
            base_path, download_config=self.download_config)

    data_files = DataFilesDict.from_patterns(
        patterns,
        base_path=base_path,
        allowed_extensions=ALL_ALLOWED_EXTENSIONS,
        download_config=self.download_config,
    )
    module_name, default_builder_kwargs = infer_module_for_data_files(
        data_files=data_files,
        path=self.name,
        download_config=self.download_config,
    )
    data_files = data_files.filter_extensions(
        _MODULE_TO_EXTENSIONS[module_name])
    # Collect metadata files if the module supports them
    supports_metadata = module_name in _MODULE_SUPPORTS_METADATA
    if self.data_files is None and supports_metadata:
        try:
            metadata_patterns = get_metadata_patterns(
                base_path, download_config=self.download_config)
        except FileNotFoundError:
            metadata_patterns = None
        if metadata_patterns is not None:
            metadata_data_files_list = DataFilesList.from_patterns(
                metadata_patterns,
                download_config=self.download_config,
                base_path=base_path)
            if metadata_data_files_list:
                data_files = DataFilesDict({
                    split: data_files_list + metadata_data_files_list
                    for split, data_files_list in data_files.items()
                })

    module_path, _ = _PACKAGED_DATASETS_MODULES[module_name]

    if metadata_configs:
        builder_configs, default_config_name = create_builder_configs_from_metadata_configs(
            module_path,
            metadata_configs,
            base_path=base_path,
            supports_metadata=supports_metadata,
            default_builder_kwargs=default_builder_kwargs,
            download_config=self.download_config,
        )
    else:
        builder_configs: List[BuilderConfig] = [
            import_main_class(module_path).BUILDER_CONFIG_CLASS(
                data_files=data_files,
                **default_builder_kwargs,
            )
        ]
        default_config_name = None
    builder_kwargs = {
        # "base_path": hf_hub_url(self.name, "", revision=revision).rstrip("/"),
        'base_path':
        _ms_api.get_file_base_path(
            namespace=_namespace,
            dataset_name=_dataset_name,
        ),
        'repo_id':
        self.name,
        'dataset_name':
        camelcase_to_snakecase(Path(self.name).name),
        'data_files': data_files,
    }
    download_config = self.download_config.copy()
    if download_config.download_desc is None:
        download_config.download_desc = 'Downloading metadata'

    # Note: `dataset_infos.json` is deprecated and can cause an error during loading if it exists

    if default_config_name is None and len(dataset_infos) == 1:
        default_config_name = next(iter(dataset_infos))

    hash = revision
    return DatasetModule(
        module_path,
        hash,
        builder_kwargs,
        dataset_infos=dataset_infos,
        builder_configs_parameters=BuilderConfigsParameters(
            metadata_configs=metadata_configs,
            builder_configs=builder_configs,
            default_config_name=default_config_name,
        ),
    )


def _download_additional_modules(
        name: str,
        dataset_name: str,
        namespace: str,
        revision: str,
        imports: Tuple[str, str, str, str],
        download_config: Optional[DownloadConfig]
) -> List[Tuple[str, str]]:
    """
    Download additional module for a module <name>.py at URL (or local path) <base_path>/<name>.py
    The imports must have been parsed first using ``get_imports``.

    If some modules need to be installed with pip, an error is raised showing how to install them.
    This function return the list of downloaded modules as tuples (import_name, module_file_path).

    The downloaded modules can then be moved into an importable directory
    with ``_copy_script_and_other_resources_in_importable_dir``.
    """
    local_imports = []
    library_imports = []
    download_config = download_config.copy()
    if download_config.download_desc is None:
        download_config.download_desc = 'Downloading extra modules'
    for import_type, import_name, import_path, sub_directory in imports:
        if import_type == 'library':
            library_imports.append((import_name, import_path))  # Import from a library
            continue

        if import_name == name:
            raise ValueError(
                f'Error in the {name} script, importing relative {import_name} module '
                f'but {import_name} is the name of the script. '
                f"Please change relative import {import_name} to another name and add a '# From: URL_OR_PATH' "
                f'comment pointing to the original relative import file path.'
            )
        if import_type == 'internal':
            _api = HubApi()
            # url_or_filename = url_or_path_join(base_path, import_path + ".py")
            file_name = import_path + '.py'
            url_or_filename = _api.get_dataset_file_url(file_name=file_name,
                                                        dataset_name=dataset_name,
                                                        namespace=namespace,
                                                        revision=revision,)
        elif import_type == 'external':
            url_or_filename = import_path
        else:
            raise ValueError('Wrong import_type')

        local_import_path = cached_path(
            url_or_filename,
            download_config=download_config,
        )
        if sub_directory is not None:
            local_import_path = os.path.join(local_import_path, sub_directory)
        local_imports.append((import_name, local_import_path))

    # Check library imports
    needs_to_be_installed = {}
    for library_import_name, library_import_path in library_imports:
        try:
            lib = importlib.import_module(library_import_name)  # noqa F841
        except ImportError:
            if library_import_name not in needs_to_be_installed or library_import_path != library_import_name:
                needs_to_be_installed[library_import_name] = library_import_path
    if needs_to_be_installed:
        _dependencies_str = 'dependencies' if len(needs_to_be_installed) > 1 else 'dependency'
        _them_str = 'them' if len(needs_to_be_installed) > 1 else 'it'
        if 'sklearn' in needs_to_be_installed.keys():
            needs_to_be_installed['sklearn'] = 'scikit-learn'
        if 'Bio' in needs_to_be_installed.keys():
            needs_to_be_installed['Bio'] = 'biopython'
        raise ImportError(
            f'To be able to use {name}, you need to install the following {_dependencies_str}: '
            f"{', '.join(needs_to_be_installed)}.\nPlease install {_them_str} using 'pip install "
            f"{' '.join(needs_to_be_installed.values())}' for instance."
        )
    return local_imports


def get_module_with_script(self) -> DatasetModule:
    if config.HF_DATASETS_TRUST_REMOTE_CODE and self.trust_remote_code is None:
        warnings.warn(
            f'The repository for {self.name} contains custom code which must be executed to correctly '
            f'load the dataset. You can inspect the repository content at https://hf.co/datasets/{self.name}\n'
            f'You can avoid this message in future by passing the argument `trust_remote_code=True`.\n'
            f'Passing `trust_remote_code=True` will be mandatory '
            f'to load this dataset from the next major release of `datasets`.',
            FutureWarning,
        )
    # get script and other files
    # local_path = self.download_loading_script()
    # dataset_infos_path = self.download_dataset_infos_file()
    # dataset_readme_path = self.download_dataset_readme_file()

    _api = HubApi()
    _dataset_name: str = self.name.split('/')[-1]
    _namespace: str = self.name.split('/')[0]

    script_file_name = f'{_dataset_name}.py'
    script_url: str = _api.get_dataset_file_url(
        file_name=script_file_name,
        dataset_name=_dataset_name,
        namespace=_namespace,
        revision=self.revision,
        extension_filter=False,
    )
    local_script_path = cached_path(
        url_or_filename=script_url, download_config=self.download_config)

    dataset_infos_path = None
    # try:
    #     dataset_infos_url: str = _api.get_dataset_file_url(
    #         file_name='dataset_infos.json',
    #         dataset_name=_dataset_name,
    #         namespace=_namespace,
    #         revision=self.revision,
    #         extension_filter=False,
    #     )
    #     dataset_infos_path = cached_path(
    #         url_or_filename=dataset_infos_url, download_config=self.download_config)
    # except Exception as e:
    #     logger.info(f'Cannot find dataset_infos.json: {e}')
    #     dataset_infos_path = None

    dataset_readme_url: str = _api.get_dataset_file_url(
        file_name='README.md',
        dataset_name=_dataset_name,
        namespace=_namespace,
        revision=self.revision,
        extension_filter=False,
    )
    dataset_readme_path = cached_path(
        url_or_filename=dataset_readme_url, download_config=self.download_config)

    imports = get_imports(local_script_path)
    local_imports = _download_additional_modules(
        name=self.name,
        dataset_name=_dataset_name,
        namespace=_namespace,
        revision=self.revision,
        imports=imports,
        download_config=self.download_config,
    )
    additional_files = []
    if dataset_infos_path:
        additional_files.append((config.DATASETDICT_INFOS_FILENAME, dataset_infos_path))
    if dataset_readme_path:
        additional_files.append((config.REPOCARD_FILENAME, dataset_readme_path))
    # copy the script and the files in an importable directory
    dynamic_modules_path = self.dynamic_modules_path if self.dynamic_modules_path else init_dynamic_modules()
    hash = files_to_hash([local_script_path] + [loc[1] for loc in local_imports])
    importable_file_path = _get_importable_file_path(
        dynamic_modules_path=dynamic_modules_path,
        module_namespace='datasets',
        subdirectory_name=hash,
        name=self.name,
    )
    if not os.path.exists(importable_file_path):
        trust_remote_code = resolve_trust_remote_code(self.trust_remote_code, self.name)
        if trust_remote_code:
            _create_importable_file(
                local_path=local_script_path,
                local_imports=local_imports,
                additional_files=additional_files,
                dynamic_modules_path=dynamic_modules_path,
                module_namespace='datasets',
                subdirectory_name=hash,
                name=self.name,
                download_mode=self.download_mode,
            )
        else:
            raise ValueError(
                f'Loading {self.name} requires you to execute the dataset script in that'
                ' repo on your local machine. Make sure you have read the code there to avoid malicious use, then'
                ' set the option `trust_remote_code=True` to remove this error.'
            )
    module_path, hash = _load_importable_file(
        dynamic_modules_path=dynamic_modules_path,
        module_namespace='datasets',
        subdirectory_name=hash,
        name=self.name,
    )
    # make the new module to be noticed by the import system
    importlib.invalidate_caches()
    builder_kwargs = {
        # "base_path": hf_hub_url(self.name, "", revision=self.revision).rstrip("/"),
        'base_path': _api.get_file_base_path(namespace=_namespace, dataset_name=_dataset_name),
        'repo_id': self.name,
    }
    return DatasetModule(module_path, hash, builder_kwargs)


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
        ignore_verifications='deprecated',
        keep_in_memory: Optional[bool] = None,
        save_infos: bool = False,
        revision: Optional[Union[str, Version]] = None,
        token: Optional[Union[bool, str]] = None,
        use_auth_token='deprecated',
        task='deprecated',
        streaming: bool = False,
        num_proc: Optional[int] = None,
        storage_options: Optional[Dict] = None,
        trust_remote_code: bool = None,
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
        if ignore_verifications != 'deprecated':
            verification_mode = VerificationMode.NO_CHECKS if ignore_verifications else VerificationMode.ALL_CHECKS
            warnings.warn(
                "'ignore_verifications' was deprecated in favor of 'verification_mode' "
                'in version 2.9.1 and will be removed in 3.0.0.\n'
                f"You can remove this warning by passing 'verification_mode={verification_mode.value}' instead.",
                FutureWarning,
            )
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
        if Path(path, config.DATASET_STATE_JSON_FILENAME).exists(
        ):
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

        # Create a dataset builder
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

        # Note: Only for preview mode
        if dataset_info_only:
            ret_dict = {}
            # Get dataset config info from python script
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

        # Return iterable dataset in case of streaming
        if streaming:
            return builder_instance.as_streaming_dataset(split=split)

        # Some datasets are already processed on the HF google storage
        # Don't try downloading from Google storage for the packaged datasets as text, json, csv or pandas
        # try_from_hf_gcs = path not in _PACKAGED_DATASETS_MODULES

        # Download and prepare data
        builder_instance.download_and_prepare(
            download_config=download_config,
            download_mode=download_mode,
            verification_mode=verification_mode,
            try_from_hf_gcs=False,
            num_proc=num_proc,
            storage_options=storage_options,
            # base_path=builder_instance.base_path,
            # file_format=builder_instance.name or 'arrow',
        )

        # Build dataset for splits
        keep_in_memory = (
            keep_in_memory if keep_in_memory is not None else is_small_dataset(
                builder_instance.info.dataset_size))
        ds = builder_instance.as_dataset(
            split=split,
            verification_mode=verification_mode,
            in_memory=keep_in_memory)
        # Rename and cast features to match task schema
        if task is not None:
            # To avoid issuing the same warning twice
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                ds = ds.prepare_for_task(task)
        if save_infos:
            builder_instance._save_infos()

        try:
            _api = HubApi()
            if is_relative_path(path) and path.count('/') == 1:
                _namespace, _dataset_name = path.split('/')
                _api.dataset_download_statistics(dataset_name=_dataset_name, namespace=_namespace)
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
        # Get dataset builder class from the processing script
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
            **builder_kwargs,  # contains base_path
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
        if download_config is None:
            download_config = DownloadConfig(**download_kwargs)
        download_config.storage_options.update({'name': subset_name})

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

        # We have several ways to get a dataset builder:
        #
        # - if path is the name of a packaged dataset module
        #   -> use the packaged module (json, csv, etc.)
        #
        # - if os.path.join(path, name) is a local python file
        #   -> use the module from the python file
        # - if path is a local directory (but no python file)
        #   -> use a packaged module (csv, text etc.) based on content of the directory
        #
        # - if path has one "/" and is dataset repository on the HF hub with a python file
        #   -> the module from the python file in the dataset repository
        # - if path has one "/" and is dataset repository on the HF hub without a python file
        #   -> use a packaged module (csv, text etc.) based on content of the repository

        # Try packaged
        if path in _PACKAGED_DATASETS_MODULES:
            return PackagedDatasetModuleFactory(
                path,
                data_dir=data_dir,
                data_files=data_files,
                download_config=download_config,
                download_mode=download_mode,
            ).get_module()
        # Try locally
        elif path.endswith(filename):
            if os.path.isfile(path):
                return LocalDatasetModuleFactoryWithScript(
                    path,
                    download_mode=download_mode,
                    dynamic_modules_path=dynamic_modules_path,
                    trust_remote_code=trust_remote_code,
                ).get_module()
            else:
                raise FileNotFoundError(
                    f"Couldn't find a dataset script at {relative_to_absolute_path(path)}"
                )
        elif os.path.isfile(combined_path):
            return LocalDatasetModuleFactoryWithScript(
                combined_path,
                download_mode=download_mode,
                dynamic_modules_path=dynamic_modules_path,
                trust_remote_code=trust_remote_code,
            ).get_module()
        elif os.path.isdir(path):
            return LocalDatasetModuleFactoryWithoutScript(
                path,
                data_dir=data_dir,
                data_files=data_files,
                download_mode=download_mode).get_module()
        # Try remotely
        elif is_relative_path(path) and path.count('/') <= 1:
            try:
                _raise_if_offline_mode_is_enabled()

                try:
                    dataset_info = HfApi().dataset_info(
                        repo_id=path,
                        revision=revision,
                        token=download_config.token,
                        timeout=100.0,
                    )
                except Exception as e:  # noqa catch any exception of hf_hub and consider that the dataset doesn't exist
                    if isinstance(
                            e,
                        (  # noqa: E131
                            OfflineModeIsEnabled,  # noqa: E131
                            requests.exceptions.
                            ConnectTimeout,  # noqa: E131, E261
                            requests.exceptions.ConnectionError,  # noqa: E131
                        ),  # noqa: E131
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
                if filename in [
                        sibling.rfilename for sibling in dataset_info.siblings
                ]:  # contains a dataset script

                    # fs = HfFileSystem(
                    #     endpoint=config.HF_ENDPOINT,
                    #     token=download_config.token)

                    # TODO
                    can_load_config_from_parquet_export = False
                    # if _require_custom_configs:
                    #     can_load_config_from_parquet_export = False
                    # elif _require_default_config_name:
                    #     with fs.open(
                    #             f'datasets/{path}/{filename}',
                    #             'r',
                    #             revision=revision,
                    #             encoding='utf-8') as f:
                    #         can_load_config_from_parquet_export = 'DEFAULT_CONFIG_NAME' not in f.read(
                    #         )
                    # else:
                    #     can_load_config_from_parquet_export = True
                    if config.USE_PARQUET_EXPORT and can_load_config_from_parquet_export:
                        # If the parquet export is ready (parquet files + info available for the current sha),
                        # we can use it instead
                        # This fails when the dataset has multiple configs and a default config and
                        # the user didn't specify a configuration name (_require_default_config_name=True).
                        try:
                            return HubDatasetModuleFactoryWithParquetExport(
                                path,
                                download_config=download_config,
                                revision=dataset_info.sha).get_module()
                        except _datasets_server.DatasetsServerError:
                            pass
                    # Otherwise we must use the dataset script if the user trusts it
                    return HubDatasetModuleFactoryWithScript(
                        path,
                        revision=revision,
                        download_config=download_config,
                        download_mode=download_mode,
                        dynamic_modules_path=dynamic_modules_path,
                        trust_remote_code=trust_remote_code,
                    ).get_module()
                else:
                    return HubDatasetModuleFactoryWithoutScript(
                        path,
                        revision=revision,
                        data_dir=data_dir,
                        data_files=data_files,
                        download_config=download_config,
                        download_mode=download_mode,
                    ).get_module()
            except Exception as e1:
                # All the attempts failed, before raising the error we should check if the module is already cached
                try:
                    return CachedDatasetModuleFactory(
                        path,
                        dynamic_modules_path=dynamic_modules_path,
                        cache_dir=cache_dir).get_module()
                except Exception:
                    # If it's not in the cache, then it doesn't exist.
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


@contextlib.contextmanager
def load_dataset_with_ctx(*args, **kwargs):
    hf_endpoint_origin = config.HF_ENDPOINT
    get_from_cache_origin = file_utils.get_from_cache
    _download_origin = DownloadManager._download
    dataset_info_origin = HfApi.dataset_info
    list_repo_tree_origin = HfApi.list_repo_tree
    get_paths_info_origin = HfApi.get_paths_info
    resolve_pattern_origin = data_files.resolve_pattern
    get_module_without_script_origin = HubDatasetModuleFactoryWithoutScript.get_module
    get_module_with_script_origin = HubDatasetModuleFactoryWithScript.get_module

    config.HF_ENDPOINT = get_endpoint()
    file_utils.get_from_cache = get_from_cache_ms
    DownloadManager._download = _download_ms
    HfApi.dataset_info = _dataset_info
    HfApi.list_repo_tree = _list_repo_tree
    HfApi.get_paths_info = _get_paths_info
    data_files.resolve_pattern = _resolve_pattern
    HubDatasetModuleFactoryWithoutScript.get_module = get_module_without_script
    HubDatasetModuleFactoryWithScript.get_module = get_module_with_script

    try:
        dataset_res = DatasetsWrapperHF.load_dataset(*args, **kwargs)
        yield dataset_res
    finally:
        config.HF_ENDPOINT = hf_endpoint_origin
        file_utils.get_from_cache = get_from_cache_origin
        DownloadManager._download = _download_origin
        HfApi.dataset_info = dataset_info_origin
        HfApi.list_repo_tree = list_repo_tree_origin
        HfApi.get_paths_info = get_paths_info_origin
        data_files.resolve_pattern = resolve_pattern_origin
        HubDatasetModuleFactoryWithoutScript.get_module = get_module_without_script_origin
        HubDatasetModuleFactoryWithScript.get_module = get_module_with_script_origin

        logger.info('Context manager of ms-dataset exited.')
