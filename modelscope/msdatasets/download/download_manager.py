# Copyright (c) Alibaba, Inc. and its affiliates.

from datasets.download.download_manager import DownloadManager
from datasets.download.streaming_download_manager import \
    StreamingDownloadManager
from datasets.utils.file_utils import cached_path, is_relative_path

from modelscope.msdatasets.download.download_config import DataDownloadConfig
from modelscope.msdatasets.utils.oss_utils import OssUtilities


class DataDownloadManager(DownloadManager):

    def __init__(self, download_config: DataDownloadConfig):
        super().__init__(
            dataset_name=download_config.dataset_name,
            data_dir=download_config.data_dir,
            download_config=download_config,
            record_checksums=True)

    def _download(self, url_or_filename: str,
                  download_config: DataDownloadConfig) -> str:
        url_or_filename = str(url_or_filename)

        oss_utilities = OssUtilities(
            oss_config=download_config.oss_config,
            dataset_name=download_config.dataset_name,
            namespace=download_config.namespace,
            revision=download_config.version)

        if is_relative_path(url_or_filename):
            # fetch oss files
            return oss_utilities.download(
                url_or_filename, download_config=download_config)
        else:
            return cached_path(
                url_or_filename, download_config=download_config)


class DataStreamingDownloadManager(StreamingDownloadManager):
    """The data streaming download manager."""

    def __init__(self, download_config: DataDownloadConfig):
        super().__init__(
            dataset_name=download_config.dataset_name,
            data_dir=download_config.data_dir,
            download_config=download_config,
            base_path=download_config.cache_dir)

    def _download(self, url_or_filename: str) -> str:
        url_or_filename = str(url_or_filename)
        oss_utilities = OssUtilities(
            oss_config=self.download_config.oss_config,
            dataset_name=self.download_config.dataset_name,
            namespace=self.download_config.namespace,
            revision=self.download_config.version)

        if is_relative_path(url_or_filename):
            # fetch oss files
            return oss_utilities.download(
                url_or_filename, download_config=self.download_config)
        else:
            return cached_path(
                url_or_filename, download_config=self.download_config)
