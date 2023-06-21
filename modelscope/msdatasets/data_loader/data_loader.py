# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from abc import ABC, abstractmethod
from typing import Optional, Union

from datasets import (Dataset, DatasetBuilder, DatasetDict, IterableDataset,
                      IterableDatasetDict)
from datasets import load_dataset as hf_load_dataset

from modelscope.hub.api import ModelScopeConfig
from modelscope.msdatasets.auth.auth_config import OssAuthConfig
from modelscope.msdatasets.context.dataset_context_config import \
    DatasetContextConfig
from modelscope.msdatasets.data_files.data_files_manager import \
    DataFilesManager
from modelscope.msdatasets.dataset_cls import ExternalDataset
from modelscope.msdatasets.meta.data_meta_manager import DataMetaManager
from modelscope.utils.constant import (DatasetFormations, DatasetPathName,
                                       DownloadMode, VirgoDatasetConfig)
from modelscope.utils.logger import get_logger
from modelscope.utils.url_utils import valid_url

logger = get_logger()


class BaseDownloader(ABC):
    """Base dataset downloader to load data."""

    def __init__(self, dataset_context_config: DatasetContextConfig):
        self.dataset_context_config = dataset_context_config

    @abstractmethod
    def process(self):
        """The entity processing pipeline for fetching the data. """
        raise NotImplementedError(
            f'No default implementation provided for {BaseDownloader.__name__}.process.'
        )

    @abstractmethod
    def _authorize(self):
        raise NotImplementedError(
            f'No default implementation provided for {BaseDownloader.__name__}._authorize.'
        )

    @abstractmethod
    def _build(self):
        raise NotImplementedError(
            f'No default implementation provided for {BaseDownloader.__name__}._build.'
        )

    @abstractmethod
    def _prepare_and_download(self):
        raise NotImplementedError(
            f'No default implementation provided for {BaseDownloader.__name__}._prepare_and_download.'
        )

    @abstractmethod
    def _post_process(self):
        raise NotImplementedError(
            f'No default implementation provided for {BaseDownloader.__name__}._post_process.'
        )


class OssDownloader(BaseDownloader):

    def __init__(self, dataset_context_config: DatasetContextConfig):
        super().__init__(dataset_context_config)

        self.data_files_builder: Optional[DataFilesManager] = None
        self.dataset: Optional[Union[Dataset, IterableDataset, DatasetDict,
                                     IterableDatasetDict,
                                     ExternalDataset]] = None
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
            self.dataset = hf_load_dataset(
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
        if isinstance(self.dataset, ExternalDataset):
            self.dataset.custom_map = self.dataset_context_config.data_meta_config.meta_type_map


class VirgoDownloader(BaseDownloader):
    """Data downloader for Virgo data source."""

    def __init__(self, dataset_context_config: DatasetContextConfig):
        super().__init__(dataset_context_config)
        self.dataset = None

    def process(self):
        """
        Sequential data fetching virgo dataset process: authorize -> build -> prepare_and_download -> post_process
        """
        self._authorize()
        self._build()
        self._prepare_and_download()
        self._post_process()

    def _authorize(self):
        """Authorization of virgo dataset."""
        from modelscope.msdatasets.auth.auth_config import VirgoAuthConfig

        cookies = ModelScopeConfig.get_cookies()
        user_info = ModelScopeConfig.get_user_info()

        if not self.dataset_context_config.auth_config:
            auth_config = VirgoAuthConfig(
                cookies=cookies, git_token='', user_info=user_info)
        else:
            auth_config = self.dataset_context_config.auth_config
            auth_config.cookies = cookies
            auth_config.git_token = ''
            auth_config.user_info = user_info

        self.dataset_context_config.auth_config = auth_config

    def _build(self):
        """
        Fetch virgo meta and build virgo dataset.
        """
        from modelscope.msdatasets.dataset_cls.dataset import VirgoDataset
        import pandas as pd

        meta_manager = DataMetaManager(self.dataset_context_config)
        meta_manager.fetch_virgo_meta()
        self.dataset_context_config = meta_manager.dataset_context_config
        self.dataset = VirgoDataset(
            **self.dataset_context_config.config_kwargs)

        virgo_cache_dir = os.path.join(
            self.dataset_context_config.cache_root_dir,
            self.dataset_context_config.namespace,
            self.dataset_context_config.dataset_name,
            self.dataset_context_config.version)
        os.makedirs(
            os.path.join(virgo_cache_dir, DatasetPathName.META_NAME),
            exist_ok=True)
        meta_content_cache_file = os.path.join(virgo_cache_dir,
                                               DatasetPathName.META_NAME,
                                               'meta_content.csv')

        if isinstance(self.dataset.meta, pd.DataFrame):
            meta_content_df = self.dataset.meta
            meta_content_df.to_csv(meta_content_cache_file, index=False)
            self.dataset.meta_content_cache_file = meta_content_cache_file
            self.dataset.virgo_cache_dir = virgo_cache_dir
            logger.info(
                f'Virgo meta content saved to {meta_content_cache_file}')

    def _prepare_and_download(self):
        """
        Fetch data-files from oss-urls in the virgo meta content.
        """

        download_virgo_files = self.dataset_context_config.config_kwargs.pop(
            'download_virgo_files', '')

        if self.dataset.data_type == 0 and download_virgo_files:
            import requests
            import json
            import shutil
            from urllib.parse import urlparse
            from functools import partial

            def download_file(meta_info_val, data_dir):
                file_url_list = []
                file_path_list = []
                try:
                    meta_info_val = json.loads(meta_info_val)
                    # get url first, if not exist, try to get inner_url
                    file_url = meta_info_val.get('url', '')
                    if file_url:
                        file_url_list.append(file_url)
                    else:
                        tmp_inner_member_list = meta_info_val.get(
                            'inner_url', '')
                        for item in tmp_inner_member_list:
                            file_url = item.get('url', '')
                            if file_url:
                                file_url_list.append(file_url)

                    for one_file_url in file_url_list:
                        is_url = valid_url(one_file_url)
                        if is_url:
                            url_parse_res = urlparse(file_url)
                            file_name = os.path.basename(url_parse_res.path)
                        else:
                            raise ValueError(f'Unsupported url: {file_url}')
                        file_path = os.path.join(data_dir, file_name)
                        file_path_list.append((one_file_url, file_path))

                except Exception as e:
                    logger.error(f'parse virgo meta info error: {e}')
                    file_path_list = []

                for file_url_item, file_path_item in file_path_list:
                    if file_path_item and not os.path.exists(file_path_item):
                        logger.info(f'Downloading file to {file_path_item}')
                        os.makedirs(data_dir, exist_ok=True)
                        with open(file_path_item, 'wb') as f:
                            f.write(requests.get(file_url_item).content)

                return file_path_list

            self.dataset.download_virgo_files = True
            download_mode = self.dataset_context_config.download_mode
            data_files_dir = os.path.join(self.dataset.virgo_cache_dir,
                                          DatasetPathName.DATA_FILES_NAME)

            if download_mode == DownloadMode.FORCE_REDOWNLOAD:
                shutil.rmtree(data_files_dir, ignore_errors=True)

            from tqdm import tqdm
            tqdm.pandas(desc='apply download_file')
            self.dataset.meta[
                VirgoDatasetConfig.
                col_cache_file] = self.dataset.meta.progress_apply(
                    lambda row: partial(
                        download_file, data_dir=data_files_dir)(row.meta_info),
                    axis=1)

    def _post_process(self):
        ...


class MaxComputeDownloader(BaseDownloader):
    """Data downloader for MaxCompute data source."""

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
