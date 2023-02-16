# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import shutil
from collections import defaultdict

import json
from datasets.utils.filelock import FileLock

from modelscope.hub.api import HubApi
from modelscope.msdatasets.context.dataset_context_config import \
    DatasetContextConfig
from modelscope.msdatasets.meta.data_meta_config import DataMetaConfig
from modelscope.msdatasets.utils.dataset_utils import (
    get_dataset_files, get_target_dataset_structure)
from modelscope.utils.constant import (DatasetFormations, DatasetPathName,
                                       DownloadMode)


class DataMetaManager(object):
    """Data-meta manager."""

    def __init__(self, dataset_context_config: DatasetContextConfig):
        self.dataset_context_config = dataset_context_config
        self.api = HubApi()

    def fetch_meta_files(self) -> None:

        # Init meta infos
        dataset_name = self.dataset_context_config.dataset_name
        namespace = self.dataset_context_config.namespace
        download_mode = self.dataset_context_config.download_mode
        version = self.dataset_context_config.version
        cache_root_dir = self.dataset_context_config.cache_root_dir
        subset_name = self.dataset_context_config.subset_name
        split = self.dataset_context_config.split

        dataset_version_cache_root_dir = os.path.join(cache_root_dir,
                                                      namespace, dataset_name,
                                                      version)
        meta_cache_dir = os.path.join(dataset_version_cache_root_dir,
                                      DatasetPathName.META_NAME)
        data_meta_config = self.dataset_context_config.data_meta_config or DataMetaConfig(
        )

        # Get lock file path
        if not subset_name:
            lock_subset_name = DatasetPathName.LOCK_FILE_NAME_ANY
        else:
            lock_subset_name = subset_name
        if not split:
            lock_split = DatasetPathName.LOCK_FILE_NAME_ANY
        else:
            lock_split = split
        lock_file_name = f'{DatasetPathName.META_NAME}{DatasetPathName.LOCK_FILE_NAME_DELIMITER}{dataset_name}' \
                         f'{DatasetPathName.LOCK_FILE_NAME_DELIMITER}{version}' \
                         f'{DatasetPathName.LOCK_FILE_NAME_DELIMITER}' \
                         f'{lock_subset_name}{DatasetPathName.LOCK_FILE_NAME_DELIMITER}{lock_split}.lock'
        lock_file_path = os.path.join(dataset_version_cache_root_dir,
                                      lock_file_name)
        os.makedirs(dataset_version_cache_root_dir, exist_ok=True)

        # Fetch meta from cache or hub if reuse dataset
        if download_mode == DownloadMode.REUSE_DATASET_IF_EXISTS:
            if os.path.exists(meta_cache_dir) and os.listdir(meta_cache_dir):
                dataset_scripts, dataset_formation = self._fetch_meta_from_cache(
                    meta_cache_dir)
            else:
                # Fetch meta-files from modelscope-hub if cache does not exist
                with FileLock(lock_file=lock_file_path):
                    os.makedirs(meta_cache_dir, exist_ok=True)
                    dataset_scripts, dataset_formation = self._fetch_meta_from_hub(
                        dataset_name, namespace, version, meta_cache_dir)
        # Fetch meta from hub if force download
        elif download_mode == DownloadMode.FORCE_REDOWNLOAD:
            # Clean meta-files
            if os.path.exists(meta_cache_dir) and os.listdir(meta_cache_dir):
                shutil.rmtree(meta_cache_dir)
            # Re-download meta-files
            with FileLock(lock_file=lock_file_path):
                os.makedirs(meta_cache_dir, exist_ok=True)
                dataset_scripts, dataset_formation = self._fetch_meta_from_hub(
                    dataset_name, namespace, version, meta_cache_dir)
        else:
            raise ValueError(
                f'Expected values of download_mode: '
                f'{DownloadMode.REUSE_DATASET_IF_EXISTS.value} or '
                f'{DownloadMode.FORCE_REDOWNLOAD.value}, but got {download_mode} .'
            )

        # Set data_meta_config
        data_meta_config.meta_cache_dir = meta_cache_dir
        data_meta_config.dataset_scripts = dataset_scripts
        data_meta_config.dataset_formation = dataset_formation

        # Set dataset_context_config
        self.dataset_context_config.data_meta_config = data_meta_config
        self.dataset_context_config.dataset_version_cache_root_dir = dataset_version_cache_root_dir
        self.dataset_context_config.global_meta_lock_file_path = lock_file_path

    def parse_dataset_structure(self):
        # Get dataset_name.json
        dataset_name = self.dataset_context_config.dataset_name
        subset_name = self.dataset_context_config.subset_name
        split = self.dataset_context_config.split
        namespace = self.dataset_context_config.namespace
        version = self.dataset_context_config.version
        data_meta_config = self.dataset_context_config.data_meta_config or DataMetaConfig(
        )

        dataset_json = None
        dataset_py_script = None
        dataset_scripts = data_meta_config.dataset_scripts
        if not dataset_scripts or len(dataset_scripts) == 0:
            raise 'Cannot find dataset meta-files, please fetch meta from modelscope hub.'
        if '.py' in dataset_scripts:
            dataset_py_script = dataset_scripts['.py'][0]
        for json_path in dataset_scripts['.json']:
            if json_path.endswith(f'{dataset_name}.json'):
                with open(json_path, encoding='utf-8') as dataset_json_file:
                    dataset_json = json.load(dataset_json_file)
                break
        if not dataset_json and not dataset_py_script:
            raise f'File {dataset_name}.json and {dataset_name}.py not found, please specify at least one meta-file.'

        # Parse meta and get dataset structure
        if dataset_py_script:
            data_meta_config.dataset_py_script = dataset_py_script
        else:
            target_subset_name, target_dataset_structure = get_target_dataset_structure(
                dataset_json, subset_name, split)
            meta_map, file_map, args_map = get_dataset_files(
                target_dataset_structure, dataset_name, namespace, version)

            data_meta_config.meta_data_files = meta_map
            data_meta_config.zip_data_files = file_map
            data_meta_config.meta_args_map = args_map
            data_meta_config.target_dataset_structure = target_dataset_structure

        self.dataset_context_config.data_meta_config = data_meta_config

    def _fetch_meta_from_cache(self, meta_cache_dir):
        local_paths = defaultdict(list)
        dataset_type = None
        for meta_file_name in os.listdir(meta_cache_dir):
            file_ext = os.path.splitext(meta_file_name)[-1]
            if file_ext == DatasetFormations.formation_mark_ext.value:
                dataset_type = int(os.path.splitext(meta_file_name)[0])
                continue
            local_paths[file_ext].append(
                os.path.join(meta_cache_dir, meta_file_name))
        if not dataset_type:
            raise FileNotFoundError(
                f'{DatasetFormations.formation_mark_ext.value} file does not exist, '
                f'please use {DownloadMode.FORCE_REDOWNLOAD.value} .')

        return local_paths, DatasetFormations(dataset_type)

    def _fetch_meta_from_hub(self, dataset_name: str, namespace: str,
                             revision: str, meta_cache_dir: str):

        # Fetch id and type of dataset
        dataset_id, dataset_type = self.api.get_dataset_id_and_type(
            dataset_name, namespace)

        # Fetch meta file-list of dataset
        file_list = self.api.get_dataset_meta_file_list(
            dataset_name, namespace, dataset_id, revision)

        # Fetch urls of meta-files
        local_paths, dataset_formation = self.api.get_dataset_meta_files_local_paths(
            dataset_name, namespace, revision, meta_cache_dir, dataset_type,
            file_list)

        return local_paths, dataset_formation
