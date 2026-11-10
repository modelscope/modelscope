# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
from dataclasses import dataclass, field
from typing import Optional, Union

from datasets.download.download_config import DownloadConfig


@dataclass
class DataDownloadConfig(DownloadConfig):
    """
    Extends `DownloadConfig` with additional attributes for data download.
    """

    dataset_name: Optional[str] = None
    namespace: Optional[str] = None
    version: Optional[str] = None
    split: Optional[Union[str, list]] = None
    data_dir: Optional[str] = None
    oss_config: Optional[dict] = field(default_factory=dict)
    meta_args_map: Optional[dict] = field(default_factory=dict)
    num_proc: int = 4

    def copy(self) -> 'DataDownloadConfig':
        return copy.deepcopy(self)
