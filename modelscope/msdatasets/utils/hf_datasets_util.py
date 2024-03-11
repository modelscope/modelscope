# noqa: isort:skip_file, yapf: disable
# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.

import os
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

import json
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
from datasets.info import DatasetInfo, DatasetInfosDict
from datasets.load import (
    ALL_ALLOWED_EXTENSIONS, BuilderConfigsParameters,
    CachedDatasetModuleFactory, DatasetModule,
    HubDatasetModuleFactoryWithoutScript,
    HubDatasetModuleFactoryWithParquetExport,
    HubDatasetModuleFactoryWithScript, LocalDatasetModuleFactoryWithoutScript,
    LocalDatasetModuleFactoryWithScript, PackagedDatasetModuleFactory,
    create_builder_configs_from_metadata_configs, get_dataset_builder_class,
    import_main_class, infer_module_for_data_files)
from datasets.naming import camelcase_to_snakecase
from datasets.packaged_modules import (_EXTENSION_TO_MODULE,
                                       _MODULE_SUPPORTS_METADATA,
                                       _MODULE_TO_EXTENSIONS,
                                       _PACKAGED_DATASETS_MODULES)
from datasets.utils import _datasets_server
from datasets.utils.file_utils import (OfflineModeIsEnabled,
                                       _raise_if_offline_mode_is_enabled,
                                       cached_path, is_local_path,
                                       is_relative_path,
                                       relative_to_absolute_path)
from datasets.utils.info_utils import is_small_dataset
from datasets.utils.metadata import MetadataConfigs
from datasets.utils.track import tracked_str
from fsspec import filesystem
from fsspec.core import _un_chain
from fsspec.utils import stringify_path
from huggingface_hub import (DatasetCard, DatasetCardData,
                             HfFileSystem, get_session)
from huggingface_hub.hf_api import DatasetInfo as HfDatasetInfo
from huggingface_hub.hf_api import HfApi, RepoFile, RepoFolder
from huggingface_hub.utils import hf_raise_for_status
from packaging import version

from modelscope import HubApi
from modelscope.hub.utils.utils import get_endpoint
from modelscope.utils.logger import get_logger

logger = get_logger()

config.HF_ENDPOINT = get_endpoint()


def _ms_download(self, url_or_filename: str,
                 download_config: DownloadConfig) -> str:
    url_or_filename = str(url_or_filename)
    if url_or_filename.startswith('hf://'):
        url_or_filename = url_or_filename.split('@')[-1].split('/', 1)[-1]
    if is_relative_path(url_or_filename):
        # append the relative path to the base_path
        # url_or_filename = url_or_path_join(self._base_path, url_or_filename)  # TODO: VERIFY
        url_or_filename = self._base_path + url_or_filename
    out = cached_path(url_or_filename, download_config=download_config)
    out = tracked_str(out)
    out.set_origin(url_or_filename)
    return out


DownloadManager._download = _ms_download


def _download(self, url_or_filename: str,
              download_config: DownloadConfig) -> str:
    url_or_filename = str(url_or_filename)
    if url_or_filename.startswith('hf://'):
        url_or_filename = url_or_filename.split('@')[-1].split('/', 1)[-1]
    if is_relative_path(url_or_filename):
        # append the relative path to the base_path
        # url_or_filename = url_or_path_join(self._base_path, url_or_filename)  # TODO: VERIFY
        url_or_filename = self._base_path + url_or_filename
    out = cached_path(url_or_filename, download_config=download_config)
    out = tracked_str(out)
    out.set_origin(url_or_filename)
    return out


DownloadManager._download = _download


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
    # TODO: 问题： self.endpoint 是huggingface.co，当加载 aya_dataset_mini时
    _endpoint = get_endpoint()
    headers = self._build_hf_headers(token=token)
    _ms_api = HubApi()
    _namespace, _dataset_name = repo_id.split('/')
    dataset_hub_id, dataset_type = _ms_api.get_dataset_id_and_type(
        dataset_name=_dataset_name, namespace=_namespace)

    revision: str = revision or 'master'
    path = f'{_endpoint}/api/v1/datasets/{dataset_hub_id}/repo/tree'
    params = {'Revision': revision, 'Root': None, 'Recursive': 'True'}

    if files_metadata:
        params['blobs'] = True

    r = get_session().get(
        path, headers=headers, timeout=timeout, params=params)
    hf_raise_for_status(r)
    data = r.json()

    # Parse data
    data_d: dict = data['Data']
    data_file_list: list = data_d['Files']
    # commit_info: dict = data_d['LatestCommitter']

    # Update data
    data['id'] = repo_id
    data['private'] = False  # TODO
    data['author'] = repo_id.split('/')[0] if repo_id else None
    data['sha'] = revision  # TODO
    data['lastModified'] = None  # TODO
    data['gated'] = False
    data['disabled'] = False
    data['downloads'] = 0  # TODO
    data['likes'] = 0  # TODO
    data['tags'] = []  # TODO
    data['cardData'] = []  # TODO
    data['createdAt'] = None  # TODO

    # {'rfilename': 'xxx', 'blobId': 'xxx', 'size': 0, 'lfs': {'size': 0, 'sha256': 'xxx', 'pointerSize': 0}}
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
                'pointerSize': 0  # TODO
            }
        }
        data['siblings'].append(file_info)

    return HfDatasetInfo(**data)


HfApi.dataset_info = _dataset_info


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

    _endpoint = get_endpoint()

    _ms_api = HubApi()
    _namespace, _dataset_name = repo_id.split('/')
    dataset_hub_id, dataset_type = _ms_api.get_dataset_id_and_type(
        dataset_name=_dataset_name, namespace=_namespace)

    revision = revision or 'master'
    # recursive = 'True' if recursive else 'False'
    root_path = path_in_repo or None
    path = f'{_endpoint}/api/v1/datasets/{dataset_hub_id}/repo/tree'
    params = {'Revision': revision, 'Root': root_path, 'Recursive': 'True'}

    r = get_session().get(path, params=params)
    hf_raise_for_status(r)
    data = r.json()

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


HfApi.list_repo_tree = _list_repo_tree


# TODO： 重写get_paths_info
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

    # repo_type = repo_type or REPO_TYPE_DATASET
    # revision = quote(revision, safe="") if revision is not None else 'master'
    # headers = self._build_hf_headers(token=token)
    #
    # response = get_session().post(
    #     f"{self.endpoint}/api/{repo_type}s/{repo_id}/paths-info/{revision}",
    #     data={
    #         "paths": paths if isinstance(paths, list) else [paths],
    #         "expand": expand,
    #     },
    #     headers=headers,
    # )
    # hf_raise_for_status(response)
    # paths_info = response.json()
    # return [
    #     RepoFile(**path_info) if path_info["type"] == "file" else RepoFolder(**path_info)
    #     for path_info in paths_info
    # ]

    # TODO: RepoFile和RepoFolder中，oid or blob_id
    return [
        RepoFile(
            path='README.md',
            size=28,
            oid='154df8298fab5ecf322016157858e08cd1bccbe1',
            lfs=None,
            last_commit=None,
            security=None),
        RepoFolder(
            path='test',
            oid='f1bd19328ef39f2f0bf04210359c0195db91d50b',
            last_commit=None),
        RepoFolder(
            path='train',
            oid='c345eb39a565ec406b11c393d02dbdab28a2f0b5',
            last_commit=None)
    ]


HfApi.get_paths_info = _get_paths_info


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

    # TODO: ONLY FOR TEST
    # base_path = 'hf://datasets/wangxingjun778test/aya_dataset_mini@master'
    # print(f'>>\nbase_path: {base_path}')

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
    # TODO: protocol  hf --> ms
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


data_files.resolve_pattern = _resolve_pattern


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

    metadata_configs = MetadataConfigs.from_dataset_card_data(
        dataset_card_data)
    dataset_infos = DatasetInfosDict.from_dataset_card_data(dataset_card_data)
    # we need a set of data files to find which dataset builder to use
    # because we need to infer module name by files extensions
    if self.data_files is not None:
        patterns = sanitize_patterns(self.data_files)
    elif metadata_configs and 'data_files' in next(
            iter(metadata_configs.values())):
        patterns = sanitize_patterns(
            next(iter(metadata_configs.values()))['data_files'])
    else:
        patterns = _get_data_patterns(
            base_path, download_config=self.download_config)

    # TODO:
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

    metadata_configs = {}  # TODO: ONLY FOR TEST

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
        HubApi.get_file_base_path(
            endpoint=get_endpoint(),
            namespace=_namespace,
            dataset_name=_dataset_name,
            revision=revision),
        'repo_id':
        self.name,
        'dataset_name':
        camelcase_to_snakecase(Path(self.name).name),
    }
    download_config = self.download_config.copy()
    if download_config.download_desc is None:
        download_config.download_desc = 'Downloading metadata'
    try:
        # this file is deprecated and was created automatically in old versions of push_to_hub
        url_or_filename = _ms_api.get_dataset_file_url(
            file_name='dataset_infos.json',
            dataset_name=_dataset_name,
            namespace=_namespace,
            revision=revision,
            extension_filter=False,
        )

        dataset_infos_path = cached_path(
            url_or_filename=url_or_filename, download_config=download_config)

        with open(dataset_infos_path, encoding='utf-8') as f:
            legacy_dataset_infos = DatasetInfosDict({
                config_name: DatasetInfo.from_dict(dataset_info_dict)
                for config_name, dataset_info_dict in json.load(f).items()
            })
            if len(legacy_dataset_infos) == 1:
                # old config e.g. named "username--dataset_name"
                legacy_config_name = next(iter(legacy_dataset_infos))
                legacy_dataset_infos['default'] = legacy_dataset_infos.pop(
                    legacy_config_name)
        legacy_dataset_infos.update(dataset_infos)
        dataset_infos = legacy_dataset_infos
    except Exception as e:
        logger.warning(f'Error while loading dataset_infos.json: {e}')
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


# TODO: Others Module --> to be added
HubDatasetModuleFactoryWithoutScript.get_module = get_module_without_script


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
        ):  # TODO: config --> TBD
            raise ValueError(
                'You are trying to load a dataset that was saved using `save_to_disk`. '
                'Please use `load_from_disk` instead.')

        if streaming and num_proc is not None:
            raise NotImplementedError(
                'Loading a streaming dataset in parallel with `num_proc` is not implemented. '
                'To parallelize streaming, you can wrap the dataset with a PyTorch DataLoader '
                'using `num_workers` > 1 instead.')

        # _base_file_path = get_file_base_path(
        #     endpoint=HUB_DATASET_ENDPOINT,
        #     namespace=_namespace,
        #     dataset_name=_dataset_name,
        #     revision=revision),

        download_mode = DownloadMode(download_mode
                                     or DownloadMode.REUSE_DATASET_IF_EXISTS)
        verification_mode = VerificationMode((
            verification_mode or VerificationMode.BASIC_CHECKS
        ) if not save_infos else VerificationMode.ALL_CHECKS)

        # Create a dataset builder
        # TODO: load_dataset_builder --> TBD
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
            base_path=builder_instance.base_path,
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

        # TODO: dataset_module_factory --> TBD
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

        if download_config is None:
            download_config = DownloadConfig(**download_kwargs)
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

                # TODO: hf_api --> TBD
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

                    # TODO: HfFileSystem --> TBD
                    fs = HfFileSystem(
                        endpoint=config.HF_ENDPOINT,
                        token=download_config.token)

                    if _require_custom_configs:
                        can_load_config_from_parquet_export = False
                    elif _require_default_config_name:
                        with fs.open(
                                f'datasets/{path}/{filename}',
                                'r',
                                revision=revision,
                                encoding='utf-8') as f:
                            can_load_config_from_parquet_export = 'DEFAULT_CONFIG_NAME' not in f.read(
                            )
                    else:
                        can_load_config_from_parquet_export = True
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
                    ).get_module()  # TODO: get_module --> TBD
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


load_dataset = DatasetsWrapperHF.load_dataset
