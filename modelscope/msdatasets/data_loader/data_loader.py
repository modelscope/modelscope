# Copyright (c) Alibaba, Inc. and its affiliates.

from abc import ABC, abstractmethod
from typing import Optional, Union

from datasets import (Dataset, DatasetBuilder, DatasetDict, IterableDataset,
                      IterableDatasetDict)
from datasets import load_dataset as hf_data_loader

from modelscope.hub.api import ModelScopeConfig
from modelscope.msdatasets.auth.auth_config import OssAuthConfig
from modelscope.msdatasets.context.dataset_context_config import \
    DatasetContextConfig
from modelscope.msdatasets.data_files.data_files_manager import \
    DataFilesManager
from modelscope.msdatasets.meta.data_meta_manager import DataMetaManager
from modelscope.utils.constant import DatasetFormations


class BaseDataLoader(ABC):
    """Base dataset loader to load data."""

    def __init__(self, dataset_context_config: DatasetContextConfig):
        self.dataset_context_config = dataset_context_config

    @abstractmethod
    def process(self):
        """The entity processing pipeline for fetching the data. """
        raise NotImplementedError(
            f'No default implementation provided for {BaseDataLoader.__name__}.process.'
        )

    @abstractmethod
    def _authorize(self):
        raise NotImplementedError(
            f'No default implementation provided for {BaseDataLoader.__name__}._authorize.'
        )

    @abstractmethod
    def _build(self):
        raise NotImplementedError(
            f'No default implementation provided for {BaseDataLoader.__name__}._build.'
        )

    @abstractmethod
    def _prepare_and_download(self):
        raise NotImplementedError(
            f'No default implementation provided for {BaseDataLoader.__name__}._prepare_and_download.'
        )

    @abstractmethod
    def _post_process(self):
        raise NotImplementedError(
            f'No default implementation provided for {BaseDataLoader.__name__}._post_process.'
        )


class OssDataLoader(BaseDataLoader):

    def __init__(self, dataset_context_config: DatasetContextConfig):
        super().__init__(dataset_context_config)

        self.data_files_builder: Optional[DataFilesManager] = None
        self.dataset: Optional[Union[Dataset, IterableDataset, DatasetDict,
                                     IterableDatasetDict]] = None
        self.builder: Optional[DatasetBuilder] = None
        self.data_files_manager: Optional[DataFilesManager] = None

    def process(self) -> None:
        """ Sequential data fetching process: authorize -> build -> prepare_and_download -> post_process,
        to keep dataset_context_config updated. """

        self._authorize()
        self._build()
        self._prepare_and_download()
        self._post_process()

    def _authorize(self) -> None:
        """ Authorization of target dataset.
        Get credentials from cache and send to the modelscope-hub in the future. """
        # TODO: obtain credentials from loacl cache when available.
        cookies = ModelScopeConfig.get_cookies()
        git_token = ModelScopeConfig.get_token()
        user_info = ModelScopeConfig.get_user_info()

        if not self.dataset_context_config.auth_config:
            auth_config = OssAuthConfig(
                cookies=cookies, git_token=git_token, user_info=user_info)
        else:
            auth_config = self.dataset_context_config.auth_config
            auth_config.cookies = cookies
            auth_config.git_token = git_token
            auth_config.user_info = user_info

        self.dataset_context_config.auth_config = auth_config

    def _build(self) -> None:
        """ Sequential data files building process: build_meta -> build_data_files , to keep context_config updated. """
        # Build meta data
        meta_manager = DataMetaManager(self.dataset_context_config)
        meta_manager.fetch_meta_files()
        meta_manager.parse_dataset_structure()
        self.dataset_context_config = meta_manager.dataset_context_config

        # Build data-files manager
        self.data_files_manager = DataFilesManager(
            dataset_context_config=self.dataset_context_config)
        self.builder = self.data_files_manager.get_data_files_builder()

    def _prepare_and_download(self) -> None:
        """ Fetch data-files from modelscope dataset-hub. """
        dataset_py_script = self.dataset_context_config.data_meta_config.dataset_py_script
        dataset_formation = self.dataset_context_config.data_meta_config.dataset_formation
        dataset_name = self.dataset_context_config.dataset_name
        subset_name = self.dataset_context_config.subset_name
        version = self.dataset_context_config.version
        split = self.dataset_context_config.split
        data_dir = self.dataset_context_config.data_dir
        data_files = self.dataset_context_config.data_files
        cache_dir = self.dataset_context_config.cache_root_dir
        download_mode = self.dataset_context_config.download_mode
        input_kwargs = self.dataset_context_config.config_kwargs

        if self.builder is None and not dataset_py_script:
            raise f'meta-file: {dataset_name}.py not found on the modelscope hub.'

        if dataset_py_script and dataset_formation == DatasetFormations.hf_compatible:
            self.dataset = hf_data_loader(
                dataset_py_script,
                name=subset_name,
                revision=version,
                split=split,
                data_dir=data_dir,
                data_files=data_files,
                cache_dir=cache_dir,
                download_mode=download_mode.value,
                ignore_verifications=True,
                **input_kwargs)
        else:
            self.dataset = self.data_files_manager.fetch_data_files(
                self.builder)

    def _post_process(self) -> None:
        ...


class MaxComputeDataLoader(BaseDataLoader):
    """Data loader for MaxCompute data source."""

    # TODO: MaxCompute data source to be supported .
    def __init__(self, dataset_context_config: DatasetContextConfig):
        super().__init__(dataset_context_config)
        self.dataset = None

    def process(self):
        ...

    def _authorize(self):
        ...

    def _build(self):
        ...

    def _prepare_and_download(self):
        ...

    def _post_process(self):
        ...
