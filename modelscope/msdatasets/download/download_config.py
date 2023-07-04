# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Optional, Union

from datasets.download.download_config import DownloadConfig


class DataDownloadConfig(DownloadConfig):

    def __init__(self):
        self.dataset_name: Optional[str] = None
        self.namespace: Optional[str] = None
        self.version: Optional[str] = None
        self.split: Optional[Union[str, list]] = None
        self.data_dir: Optional[str] = None
        self.oss_config: Optional[dict] = {}
        self.meta_args_map: Optional[dict] = {}
        self.num_proc: int = 4

    def copy(self) -> 'DataDownloadConfig':
        return self
