# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Dict, Union

import datasets
import pandas as pd
import pyarrow as pa
from datasets import (ArrowBasedBuilder, GeneratorBasedBuilder,
                      IterableDataset, IterableDatasetDict)
from datasets.filesystems import is_remote_filesystem
from datasets.info import DatasetInfo
from datasets.naming import camelcase_to_snakecase
from datasets.packaged_modules import csv
from datasets.utils.filelock import FileLock
from datasets.utils.py_utils import map_nested

from modelscope.hub.api import HubApi
from modelscope.msdatasets.context.dataset_context_config import \
    DatasetContextConfig
from modelscope.msdatasets.dataset_cls.dataset import (ExternalDataset,
                                                       NativeIterableDataset)
from modelscope.msdatasets.download.download_manager import \
    DataStreamingDownloadManager
from modelscope.msdatasets.utils.dataset_utils import \
    get_subdir_hash_from_split
from modelscope.utils.constant import (DEFAULT_DATASET_NAMESPACE,
                                       DatasetPathName, DownloadMode)
from modelscope.utils.logger import get_logger

logger = get_logger()

DELIMITER_NAME = 'delimiter'
DEFAULT_CSV_DELIMITER = ','


class CsvDatasetBuilder(csv.Csv):

    def __init__(self, dataset_context_config: DatasetContextConfig):
        # Init config args
        self.dataset_name = dataset_context_config.dataset_name
        self.cache_root_dir = dataset_context_config.cache_root_dir
        self.namespace = dataset_context_config.namespace
        self.version = dataset_context_config.version
        self.subset_name = dataset_context_config.subset_name
        self.split = dataset_context_config.split
        self.meta_data_files = dataset_context_config.data_meta_config.meta_data_files
        self.zip_data_files = dataset_context_config.data_meta_config.zip_data_files
        self.input_config_kwargs = dataset_context_config.config_kwargs

        self.cache_build_dir = os.path.join(self.cache_root_dir,
                                            self.namespace, self.dataset_name,
                                            self.version,
                                            DatasetPathName.META_NAME)
        self.csv_delimiter = DEFAULT_CSV_DELIMITER
        if DELIMITER_NAME in self.input_config_kwargs:
            self.csv_delimiter = self.input_config_kwargs[DELIMITER_NAME]

        split = self.split or list(dataset_context_config.data_meta_config.
                                   target_dataset_structure.keys())
        sub_dir_hash = get_subdir_hash_from_split(
            split=split, version=self.version)

        super().__init__(
            cache_dir=self.cache_build_dir,
            config_name=self.namespace,
            hash=sub_dir_hash,
            data_files=self.meta_data_files,
            **self.input_config_kwargs)

        self.info.builder_name = self.dataset_name
        self.name = camelcase_to_snakecase(self.dataset_name)

    def _build_cache_dir(self, namespace=DEFAULT_DATASET_NAMESPACE):
        builder_data_dir = os.path.join(
            self._cache_dir_root,
            self._relative_data_dir(
                with_version=False, with_hash=True, namespace=namespace))

        return builder_data_dir

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
                file, iterator=True, dtype=dtype, delimiter=self.csv_delimiter)
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


class TaskSpecificDatasetBuilder(CsvDatasetBuilder):

    def __init__(self, dataset_context_config: DatasetContextConfig):

        # Init args
        self.name = dataset_context_config.dataset_name
        self.subset_name = dataset_context_config.subset_name
        self.namespace = dataset_context_config.namespace
        self.split = dataset_context_config.split
        self.version = dataset_context_config.version
        split = self.split or list(dataset_context_config.data_meta_config.
                                   target_dataset_structure.keys())
        self.hash = get_subdir_hash_from_split(
            split=split, version=self.version)
        self.data_files = dataset_context_config.data_meta_config.meta_data_files
        self.zip_data_files = dataset_context_config.data_meta_config.zip_data_files
        self.split_path_dict = None
        self.config = None
        self.info = DatasetInfo.from_dict(
            {'builder_name': dataset_context_config.dataset_name})
        self._cache_dir_root = os.path.expanduser(
            dataset_context_config.cache_root_dir)
        self._cache_dir = self._build_cache_dir()
        self._config_kwargs = dataset_context_config.data_meta_config.meta_args_map

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


class IterableDatasetBuilder(csv.Csv):

    def __init__(self, dataset_context_config: DatasetContextConfig):
        # Init config args
        self.dataset_name = dataset_context_config.dataset_name
        self.cache_root_dir = dataset_context_config.cache_root_dir
        self.namespace = dataset_context_config.namespace
        self.version = dataset_context_config.version
        self.subset_name = dataset_context_config.subset_name
        self.split = dataset_context_config.split
        self.meta_data_files = dataset_context_config.data_meta_config.meta_data_files
        self.zip_data_files = dataset_context_config.data_meta_config.zip_data_files
        self.input_config_kwargs = dataset_context_config.config_kwargs

        self.cache_build_dir = os.path.join(self.cache_root_dir,
                                            self.namespace, self.dataset_name,
                                            self.version,
                                            DatasetPathName.META_NAME)
        self.csv_delimiter = DEFAULT_CSV_DELIMITER
        if DELIMITER_NAME in self.input_config_kwargs:
            self.csv_delimiter = self.input_config_kwargs[DELIMITER_NAME]

        split = self.split or list(dataset_context_config.data_meta_config.
                                   target_dataset_structure.keys())
        sub_dir_hash = get_subdir_hash_from_split(
            split=split, version=self.version)

        super().__init__(
            cache_dir=self.cache_build_dir,
            config_name=self.namespace,
            hash=sub_dir_hash,
            data_files=None,  # TODO: self.meta_data_files,
            **self.input_config_kwargs)

        self.info.builder_name = self.dataset_name
        self.name = camelcase_to_snakecase(self.dataset_name)

    @staticmethod
    def get_builder_instance(
            dataset_context_config: DatasetContextConfig) -> csv.Csv:
        builder_instance = IterableDatasetBuilder(
            dataset_context_config=dataset_context_config)
        return builder_instance

    def as_streaming_dataset(
        self, dl_manager: DataStreamingDownloadManager
    ) -> Union[Dict[str, IterableDataset], IterableDataset]:

        if not isinstance(self, (GeneratorBasedBuilder, ArrowBasedBuilder)):
            raise ValueError(f'Builder {self.name} is not streamable.')

        is_local = not is_remote_filesystem(self._fs)
        if not is_local:
            raise NotImplementedError(
                f'Loading a streaming dataset cached in a {type(self._fs).__name__} is not supported yet.'
            )

        self._check_manual_download(dl_manager)
        splits_generators = {
            sg.name: sg
            for sg in self._split_generators(dl_manager)
        }

        # By default, return all splits
        split = dl_manager.download_config.split
        if split is None:
            splits_generator = splits_generators
        elif split in splits_generators:
            splits_generator = splits_generators[split]
        else:
            raise ValueError(
                f'Bad split: {split}. Available splits: {list(splits_generators)}'
            )

        # Create a dataset for each of the given splits
        streaming_datasets = map_nested(
            self._as_streaming_dataset_single,
            splits_generator,
            map_tuple=True,
        )
        if isinstance(streaming_datasets, dict):
            streaming_datasets = IterableDatasetDict(streaming_datasets)
        return streaming_datasets

    def _split_generators(self, dl_manager: DataStreamingDownloadManager):
        splits = []
        meta_data_file = ''
        zip_data_file = ''
        if self.meta_data_files:
            meta_data_file = next(iter(self.meta_data_files.values()))
        if self.zip_data_files:
            zip_data_file = next(iter(self.zip_data_files.values()))
        if meta_data_file and not zip_data_file:
            for split_name, meta_file_url in self.meta_data_files.items():
                splits.append(
                    datasets.SplitGenerator(
                        name=split_name,
                        gen_kwargs={
                            'meta': meta_file_url,
                            'files': [],
                            'dl_manager': dl_manager,
                        }))

        elif meta_data_file and zip_data_file:
            for split_name, files in self.zip_data_files.items():
                if isinstance(files, str):
                    files = [files]
                meta_file_url = self.meta_data_files.get(split_name)
                splits.append(
                    datasets.SplitGenerator(
                        name=split_name,
                        gen_kwargs={
                            'meta': meta_file_url,
                            'files': files,
                            'dl_manager': dl_manager,
                        }))

        elif not meta_data_file and zip_data_file:
            for split_name, files in self.zip_data_files.items():
                if isinstance(files, str):
                    files = [files]
                splits.append(
                    datasets.SplitGenerator(
                        name=split_name,
                        gen_kwargs={
                            'meta': '',
                            'files': files,
                            'dl_manager': dl_manager,
                        }))

        else:
            raise f'Neither column meta nor data file found in {self.dataset_name}.json, specify at least one column.'

        return splits

    def _as_streaming_dataset_single(
        self,
        splits_generator,
    ) -> NativeIterableDataset:

        ex_iterable = self._get_examples_iterable_for_split(splits_generator)
        return NativeIterableDataset(
            ex_iterable, info=self.info, split=splits_generator.name)

    def _generate_tables(self, **gen_kwargs):

        meta_file_url = gen_kwargs.get('meta')
        files = gen_kwargs.get('files')
        dl_manager = gen_kwargs.get('dl_manager')

        hub_api = HubApi()
        is_zip = False
        zip_file_name = ''

        if files:
            zip_file = str(next(iter(files)))
            if zip_file.endswith('.zip'):
                is_zip = True
                zip_file_name = os.path.splitext(zip_file)[0]

        if meta_file_url and not files:
            headers, texts = hub_api.fetch_single_csv_script(meta_file_url)
            meta_csv_mapping = IterableDatasetBuilder.trans_data_to_mapping(
                headers, texts, self.csv_delimiter)
            pa_table = pa.Table.from_pydict(meta_csv_mapping)
            yield 0, pa_table

        elif meta_file_url and files:
            # Get meta file
            headers, texts = hub_api.fetch_single_csv_script(meta_file_url)
            meta_csv_mapping = IterableDatasetBuilder.trans_data_to_mapping(
                headers, texts, self.csv_delimiter)

            if is_zip:
                oss_config_for_unzipped = hub_api.get_dataset_access_config_for_unzipped(
                    self.dataset_name, self.namespace, self.version,
                    zip_file_name)
                dl_manager.download_config.oss_config = oss_config_for_unzipped

            pa_table = pa.Table.from_pydict(meta_csv_mapping)
            yield 0, pa_table

        elif not meta_file_url and files:
            pa_table = pa.Table.from_pydict({'Input:FILE': files})
            yield 0, pa_table

        else:
            raise f'Neither column meta nor data file found in {self.dataset_name}.json .'

    @staticmethod
    def trans_data_to_mapping(headers: str, texts: list, delimiter: str):
        res = {}
        headers = headers.split(delimiter)
        for idx in range(0, len(headers)):
            col_list = []
            for line in texts:
                col_list.append(line.split(delimiter)[idx])
            res[headers[idx]] = col_list
        return res
