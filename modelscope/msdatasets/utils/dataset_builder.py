# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
import os
from typing import Mapping, Sequence, Union

import datasets
import pandas as pd
import pyarrow as pa
from datasets.info import DatasetInfo
from datasets.naming import camelcase_to_snakecase
from datasets.packaged_modules import csv
from datasets.utils.filelock import FileLock

from modelscope.utils.constant import (DEFAULT_DATASET_NAMESPACE,
                                       EXTENSIONS_TO_LOAD, DownloadMode)
from modelscope.utils.logger import get_logger

logger = get_logger()


class MsCsvDatasetBuilder(csv.Csv):

    def __init__(
        self,
        dataset_name: str,
        cache_dir: str,
        namespace: str,
        subset_name: str,
        hash: str,
        meta_data_files: Mapping[str, Union[str, Sequence[str]]],
        zip_data_files: Mapping[str, Union[str, Sequence[str]]] = None,
        **config_kwargs,
    ):
        super().__init__(
            cache_dir=cache_dir,
            name=subset_name,
            hash=hash,
            data_files=meta_data_files,
            **config_kwargs)

        self.name = camelcase_to_snakecase(dataset_name)
        self.info.builder_name = dataset_name
        self._cache_dir = self._build_cache_dir(namespace=namespace)
        lock_path = os.path.join(
            self._cache_dir_root,
            self._cache_dir.replace(os.sep, '_') + '.lock')
        with FileLock(lock_path):
            # check if data exist
            if os.path.exists(self._cache_dir):
                if len(os.listdir(self._cache_dir)) > 0:
                    logger.info(
                        f'Overwrite dataset info from restored data version, cache_dir is {self._cache_dir}'
                    )
                # dir exists but no data, remove the empty dir as data aren't available anymore
                else:
                    logger.warning(
                        f'Old caching folder {self._cache_dir} for dataset {self.name} exists '
                        f'but not data were found. Removing it. ')
                    os.rmdir(self._cache_dir)
        self.zip_data_files = zip_data_files

    def _relative_data_dir(self,
                           with_version=True,
                           with_hash=True,
                           namespace=DEFAULT_DATASET_NAMESPACE) -> str:
        """Relative path of this dataset in cache_dir:
        Will be:
            self.name/self.config.version/self.hash/
        or if a namespace has been specified:
            self.namespace___self.name/self.config.version/self.hash/
        """
        builder_data_dir = self.info.builder_name if namespace is None else f'{namespace}___{self.info.builder_name}'
        builder_config = self.config
        hash = self.hash
        if builder_config:
            builder_data_dir = os.path.join(builder_data_dir, self.config_id)
        if with_version:
            builder_data_dir = os.path.join(builder_data_dir,
                                            str(self.config.version))
        if with_hash and hash and isinstance(hash, str):
            builder_data_dir = os.path.join(builder_data_dir, hash)
        return builder_data_dir

    def _build_cache_dir(self, namespace=DEFAULT_DATASET_NAMESPACE):
        builder_data_dir = os.path.join(
            self._cache_dir_root,
            self._relative_data_dir(
                with_version=False, with_hash=True, namespace=namespace))

        return builder_data_dir

    def _split_generators(self, dl_manager):
        if not self.config.data_files:
            raise ValueError(
                'At least one data file must be specified, but got none.')
        data_files = dl_manager.download_and_extract(self.config.data_files)
        zip_data_files = dl_manager.download_and_extract(self.zip_data_files)
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            splits.append(
                datasets.SplitGenerator(
                    name=split_name,
                    gen_kwargs={
                        'files': dl_manager.iter_files(files),
                        'base_dir': zip_data_files.get(split_name)
                    }))
        return splits

    def _generate_tables(self, files, base_dir):
        schema = pa.schema(self.config.features.type
                           ) if self.config.features is not None else None
        dtype = {
            name: dtype.to_pandas_dtype()
            for name, dtype in zip(schema.names, schema.types)
        } if schema else None
        for file_idx, file in enumerate(files):
            csv_file_reader = pd.read_csv(
                file,
                iterator=True,
                dtype=dtype,
                **self.config.read_csv_kwargs)
            transform_fields = []
            for field_name in csv_file_reader._engine.names:
                if field_name.endswith(':FILE'):
                    transform_fields.append(field_name)
            try:
                for batch_idx, df in enumerate(csv_file_reader):
                    for field_name in transform_fields:
                        if base_dir:
                            df[field_name] = df[field_name].apply(
                                lambda x: os.path.join(base_dir, x))
                    pa_table = pa.Table.from_pandas(df, schema=schema)
                    yield (file_idx, batch_idx), pa_table
            except ValueError as e:
                logger.error(
                    f"Failed to read file '{file}' with error {type(e)}: {e}")
                raise


class TaskSpecificDatasetBuilder(MsCsvDatasetBuilder):

    def __init__(
        self,
        dataset_name: str,
        cache_dir: str,
        namespace: str,
        subset_name: str,
        hash: str,
        meta_data_files: Mapping[str, Union[str, Sequence[str]]],
        zip_data_files: Mapping[str, Union[str, Sequence[str]]] = None,
        **config_kwargs,
    ):
        self.name = dataset_name
        self.subset_name = subset_name
        self.namespace = namespace
        self.hash = hash
        self.data_files = meta_data_files
        self.zip_data_files = zip_data_files
        self.split_path_dict = None
        self.config = None
        self.info = DatasetInfo.from_dict({'builder_name': dataset_name})
        self._cache_dir_root = os.path.expanduser(cache_dir)
        self._cache_dir = self._build_cache_dir()
        self._config_kwargs = config_kwargs

    def download_and_prepare(self, download_mode, dl_manager,
                             **download_kwargs):
        # Prevent parallel disk operations
        lock_path = os.path.join(
            self._cache_dir_root,
            self._cache_dir.replace(os.sep, '_') + '.lock')
        with FileLock(lock_path):
            data_exists = os.path.exists(self._cache_dir)
            if data_exists and download_mode == DownloadMode.REUSE_DATASET_IF_EXISTS:
                logger.warning(
                    f'Reusing dataset {self.name} ({self._cache_dir})')
                return
            logger.info(f'Generating dataset {self.name} ({self._cache_dir})')
        self._download_and_prepare(dl_manager=dl_manager)

    def _download_and_prepare(self, dl_manager):
        self.split_path_dict = dl_manager.download_and_extract(
            self.zip_data_files)

    def as_dataset(self):
        return ExternalDataset(self.split_path_dict, self._config_kwargs)


class ExternalDataset(object):

    def __init__(self, split_path_dict, config_kwargs):
        self.split_path_dict = split_path_dict
        self.config_kwargs = copy.deepcopy(config_kwargs)
        self.config_kwargs.update({'split_config': split_path_dict})
        self.ext_dataset = None
        self.split_data_files = {k: [] for k, _ in split_path_dict.items()}
        file_ext = ''

        for split_name, split_dir in split_path_dict.items():
            if isinstance(split_dir, str) and os.path.isdir(split_dir):
                split_file_names = os.listdir(split_dir)
                set_files_exts = set([
                    os.path.splitext(file_name)[-1].strip('.')
                    for file_name in split_file_names
                ])
                if '' in set_files_exts:
                    continue
                # ensure these files have same extensions
                if len(set_files_exts) != 1:
                    supported_exts = ','.join(EXTENSIONS_TO_LOAD.keys())
                    logger.error(
                        f'Split-{split_name} has been ignored, please flatten your folder structure, '
                        f'and make sure these files have same extensions. '
                        f'Supported extensions: {supported_exts} .')
                    continue
                file_ext = list(set_files_exts)[0]
                if file_ext not in EXTENSIONS_TO_LOAD:
                    continue

                split_file_paths = [
                    os.path.join(split_dir, file_name)
                    for file_name in split_file_names
                ]
                self.split_data_files[split_name] = split_file_paths

        if file_ext:
            file_ext = EXTENSIONS_TO_LOAD.get(file_ext)
            self.ext_dataset = datasets.load_dataset(
                file_ext, data_files=self.split_data_files, **config_kwargs)

    def __len__(self):
        return len(self.split_path_dict
                   ) if not self.ext_dataset else self.ext_dataset.__len__()

    def __getitem__(self, item):
        if not self.ext_dataset:
            return self.split_path_dict.get(item)
        else:
            return self.ext_dataset.__getitem__(item)

    def __iter__(self):
        if not self.ext_dataset:
            for k, v in self.split_path_dict.items():
                yield k, v
        else:
            for k, v in self.ext_dataset.items():
                yield k, v
