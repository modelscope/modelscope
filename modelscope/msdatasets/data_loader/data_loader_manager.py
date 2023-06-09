# Copyright (c) Alibaba, Inc. and its affiliates.

import enum
import os
from abc import ABC, abstractmethod

from datasets import load_dataset as hf_data_loader

from modelscope.hub.api import HubApi
from modelscope.msdatasets.context.dataset_context_config import \
    DatasetContextConfig
from modelscope.msdatasets.data_loader.data_loader import OssDownloader
from modelscope.utils.constant import EXTENSIONS_TO_LOAD
from modelscope.utils.logger import get_logger

logger = get_logger()


class LocalDataLoaderType(enum.Enum):
    """ Supported data loader types for local dataset: huggingface, PyTorch, Tensorflow """
    HF_DATA_LOADER = 'hf_data_loader'
    TORCH_DATA_LOADER = 'torch_data_loader'
    TF_DATA_LOADER = 'tf_data_loader'


class RemoteDataLoaderType(enum.Enum):
    """ Supported data loader types for remote dataset: huggingface, modelscope """
    HF_DATA_LOADER = 'hf_data_loader'
    MS_DATA_LOADER = 'ms_data_loader'


class DataLoaderManager(ABC):
    """Data loader manager, base class."""

    def __init__(self, dataset_context_config: DatasetContextConfig):
        self.dataset_context_config = dataset_context_config

    @abstractmethod
    def load_dataset(self, data_loader_type: enum.Enum):
        ...


class LocalDataLoaderManager(DataLoaderManager):
    """Data loader manager for loading local data."""

    def __init__(self, dataset_context_config: DatasetContextConfig):
        super().__init__(dataset_context_config=dataset_context_config)

    def load_dataset(self, data_loader_type: enum.Enum):
        # Get args from context
        dataset_name = self.dataset_context_config.dataset_name
        subset_name = self.dataset_context_config.subset_name
        version = self.dataset_context_config.version
        split = self.dataset_context_config.split
        data_dir = self.dataset_context_config.data_dir
        data_files = self.dataset_context_config.data_files
        cache_root_dir = self.dataset_context_config.cache_root_dir
        download_mode = self.dataset_context_config.download_mode
        use_streaming = self.dataset_context_config.use_streaming
        input_config_kwargs = self.dataset_context_config.config_kwargs

        # load local single file
        if os.path.isfile(dataset_name):
            file_ext = os.path.splitext(dataset_name)[1].strip('.')
            if file_ext in EXTENSIONS_TO_LOAD:
                split = None
                data_files = [dataset_name]
                dataset_name = EXTENSIONS_TO_LOAD.get(file_ext)

        # Select local data loader
        # TODO: more loaders to be supported.
        if data_loader_type == LocalDataLoaderType.HF_DATA_LOADER:
            # Build huggingface data loader and return dataset.
            return hf_data_loader(
                dataset_name,
                name=subset_name,
                revision=version,
                split=split,
                data_dir=data_dir,
                data_files=data_files,
                cache_dir=cache_root_dir,
                download_mode=download_mode.value,
                streaming=use_streaming,
                ignore_verifications=True,
                **input_config_kwargs)
        raise f'Expected local data loader type: {LocalDataLoaderType.HF_DATA_LOADER.value}.'


class RemoteDataLoaderManager(DataLoaderManager):
    """Data loader manager for loading remote data."""

    def __init__(self, dataset_context_config: DatasetContextConfig):
        super().__init__(dataset_context_config=dataset_context_config)
        self.api = HubApi()

    def load_dataset(self, data_loader_type: enum.Enum):
        # Get args from context
        dataset_name = self.dataset_context_config.dataset_name
        namespace = self.dataset_context_config.namespace
        subset_name = self.dataset_context_config.subset_name
        version = self.dataset_context_config.version
        split = self.dataset_context_config.split
        data_dir = self.dataset_context_config.data_dir
        data_files = self.dataset_context_config.data_files
        download_mode_val = self.dataset_context_config.download_mode.value
        use_streaming = self.dataset_context_config.use_streaming
        input_config_kwargs = self.dataset_context_config.config_kwargs

        # To use the huggingface data loader
        if data_loader_type == RemoteDataLoaderType.HF_DATA_LOADER:
            dataset_ret = hf_data_loader(
                dataset_name,
                name=subset_name,
                revision=version,
                split=split,
                data_dir=data_dir,
                data_files=data_files,
                download_mode=download_mode_val,
                streaming=use_streaming,
                ignore_verifications=True,
                **input_config_kwargs)
            # download statistics
            self.api.dataset_download_statistics(
                dataset_name=dataset_name,
                namespace=namespace,
                use_streaming=use_streaming)
            return dataset_ret
        # To use the modelscope data loader
        elif data_loader_type == RemoteDataLoaderType.MS_DATA_LOADER:
            oss_downloader = OssDownloader(
                dataset_context_config=self.dataset_context_config)
            oss_downloader.process()
            # download statistics
            self.api.dataset_download_statistics(
                dataset_name=dataset_name,
                namespace=namespace,
                use_streaming=use_streaming)
            return oss_downloader.dataset
        else:
            raise f'Expected remote data loader type: {RemoteDataLoaderType.HF_DATA_LOADER.value}/' \
                  f'{RemoteDataLoaderType.MS_DATA_LOADER.value}, but got {data_loader_type} .'
