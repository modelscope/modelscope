# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.msdatasets.dataset_cls.custom_datasets.video_summarization_dataset import \
    VideoSummarizationDataset
from modelscope.utils.logger import get_logger

logger = get_logger()
logger.warning(
    'The reference has been Deprecated, '
    'please use `from modelscope.msdatasets.dataset_cls.'
    'custom_datasets.video_summarization_dataset import VideoSummarizationDataset`'
)
