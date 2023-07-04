# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Union

from datasets import DatasetBuilder

from modelscope.hub.api import HubApi
from modelscope.msdatasets.context.dataset_context_config import \
    DatasetContextConfig
from modelscope.msdatasets.download.dataset_builder import (
    CsvDatasetBuilder, IterableDatasetBuilder, TaskSpecificDatasetBuilder)
from modelscope.msdatasets.download.download_config import DataDownloadConfig
from modelscope.msdatasets.download.download_manager import (
    DataDownloadManager, DataStreamingDownloadManager)
from modelscope.utils.constant import (META_FILES_FORMAT, DatasetPathName,
                                       DownloadMode, MetaDataFields)


class DataFilesManager(object):
    """The modelscope data-files manager."""

    def __init__(self, dataset_context_config: DatasetContextConfig):

        # Get dataset config info
        self.dataset_name = dataset_context_config.dataset_name
        self.namespace = dataset_context_config.namespace
        self.version = dataset_context_config.version
        self.subset_name = dataset_context_config.subset_name
        self.split = dataset_context_config.split
        self.meta_data_files = dataset_context_config.data_meta_config.meta_data_files
        self.meta_args_map = dataset_context_config.data_meta_config.meta_args_map
        self.zip_data_files = dataset_context_config.data_meta_config.zip_data_files
        self.download_mode = dataset_context_config.download_mode
        self.use_streaming = dataset_context_config.use_streaming
        self.input_config_kwargs = dataset_context_config.config_kwargs

        # Get download_config
        download_config = dataset_context_config.download_config or DataDownloadConfig(
        )
        download_config.dataset_name = dataset_context_config.dataset_name
        download_config.namespace = dataset_context_config.namespace
        download_config.version = dataset_context_config.version
        download_config.split = dataset_context_config.split
        download_config.cache_dir = os.path.join(
            dataset_context_config.cache_root_dir, self.namespace,
            self.dataset_name, self.version, DatasetPathName.DATA_FILES_NAME)

        is_force_download = dataset_context_config.download_mode == DownloadMode.FORCE_REDOWNLOAD
        download_config.force_download = bool(is_force_download)
        download_config.force_extract = bool(is_force_download)
        download_config.use_etag = False

        # Get oss config
        api = HubApi()
        self.oss_config = api.get_dataset_access_config(
            self.dataset_name, self.namespace, self.version)

        # Set context. Note: no need to update context_config.
        download_config.oss_config = self.oss_config
        download_config.num_proc = self.input_config_kwargs.get('num_proc', 4)
        dataset_context_config.download_config = download_config
        self.dataset_context_config = dataset_context_config
        os.makedirs(download_config.cache_dir, exist_ok=True)

    def get_data_files_builder(self) -> Union[DatasetBuilder, None]:
        """ Build download manager. """

        if self.use_streaming:
            return IterableDatasetBuilder.get_builder_instance(
                dataset_context_config=self.dataset_context_config)

        if not self.meta_data_files:
            return None

        meta_data_file = next(iter(self.meta_data_files.values()))
        meta_args_map_file = next(iter(self.meta_args_map.values()))
        if meta_args_map_file is None:
            meta_args_map_file = {}

        if not meta_data_file or meta_args_map_file.get(
                MetaDataFields.ARGS_BIG_DATA):
            meta_args_map_file.update(self.input_config_kwargs)
            self.dataset_context_config.data_meta_config.meta_args_map = meta_args_map_file

            builder = TaskSpecificDatasetBuilder(
                dataset_context_config=self.dataset_context_config)
        elif meta_data_file and os.path.splitext(
                meta_data_file)[-1] in META_FILES_FORMAT:
            builder = CsvDatasetBuilder(
                dataset_context_config=self.dataset_context_config)
        else:
            raise NotImplementedError(
                f'Dataset meta file extensions "{os.path.splitext(meta_data_file)[-1]}" is not implemented yet'
            )
        return builder

    def fetch_data_files(self, builder):
        """ Fetch the data-files from dataset-hub. """

        if self.dataset_context_config.use_streaming:
            dl_manager = DataStreamingDownloadManager(
                download_config=self.dataset_context_config.download_config)
            return builder.as_streaming_dataset(dl_manager)
        else:

            self.dataset_context_config.download_config.meta_args_map = \
                self.dataset_context_config.data_meta_config.meta_args_map

            dl_manager = DataDownloadManager(
                download_config=self.dataset_context_config.download_config)
            builder.download_and_prepare(
                dl_manager=dl_manager,
                download_mode=self.download_mode.value,
                try_from_hf_gcs=False)
            return builder.as_dataset()
