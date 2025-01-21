# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Optional, Union

from datasets.download.download_config import DownloadConfig


class DataDownloadConfig(DownloadConfig):
    """
    Extends `DownloadConfig` with additional attributes for data download.
    """

    dataset_name: Optional[str] = None
    namespace: Optional[str] = None
    version: Optional[str] = None
    split: Optional[Union[str, list]] = None
    data_dir: Optional[str] = None
    oss_config: Optional[dict] = {}
    meta_args_map: Optional[dict] = {}
    num_proc: int = 4

    def copy(self) -> 'DataDownloadConfig':
        return self
