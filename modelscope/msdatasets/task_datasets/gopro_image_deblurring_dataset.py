# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.msdatasets.dataset_cls.custom_datasets.gopro_image_deblurring_dataset import \
    GoproImageDeblurringDataset
from modelscope.utils.logger import get_logger

logger = get_logger()
logger.warning(
    'The reference has been Deprecated, '
    'please use `from modelscope.msdatasets.dataset_cls.'
    'custom_datasets.gopro_image_deblurring_dataset import GoproImageDeblurringDataset`'
)
