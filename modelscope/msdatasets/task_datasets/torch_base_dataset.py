# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.msdatasets.dataset_cls.custom_datasets import \
    TorchCustomDataset as TorchTaskDataset
from modelscope.utils.logger import get_logger

logger = get_logger()
logger.warning(
    'The reference has been Deprecated in modelscope v1.4.0+, '
    'please use `from modelscope.msdatasets.dataset_cls.custom_datasets import TorchCustomDataset`'
)
