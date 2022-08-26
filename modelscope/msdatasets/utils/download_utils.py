from typing import Optional

from datasets.utils.download_manager import DownloadConfig, DownloadManager
from datasets.utils.file_utils import cached_path, is_relative_path

from .oss_utils import OssUtilities


class DatasetDownloadManager(DownloadManager):

    def __init__(
        self,
        dataset_name: str,
        namespace: str,
        version: str,
        data_dir: Optional[str] = None,
        download_config: Optional[DownloadConfig] = None,
        base_path: Optional[str] = None,
        record_checksums=True,
    ):
        super().__init__(dataset_name, data_dir, download_config, base_path,
                         record_checksums)
        self._namespace = namespace
        self._version = version
        from modelscope.hub.api import HubApi
        api = HubApi()
        oss_config = api.get_dataset_access_config(self._dataset_name,
                                                   self._namespace,
                                                   self._version)
        self.oss_utilities = OssUtilities(oss_config)

    def _download(self, url_or_filename: str,
                  download_config: DownloadConfig) -> str:
        url_or_filename = str(url_or_filename)
        if is_relative_path(url_or_filename):
            # fetch oss files
            return self.oss_utilities.download(
                url_or_filename, download_config=download_config)
        else:
            return cached_path(
                url_or_filename, download_config=download_config)
