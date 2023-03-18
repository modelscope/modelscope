# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .torch_base_dataset import TorchTaskDataset
    from .gopro_image_deblurring_dataset import GoproImageDeblurringDataset
    from .reds_image_deblurring_dataset import RedsImageDeblurringDataset
    from .sidd_image_denoising import SiddImageDenoisingDataset
    from .video_summarization_dataset import VideoSummarizationDataset
else:
    _import_structure = {
        'torch_base_dataset': ['TorchTaskDataset'],
        'gopro_image_deblurring_dataset': ['GoproImageDeblurringDataset'],
        'reds_image_deblurring_dataset': ['RedsImageDeblurringDataset'],
        'sidd_image_denoising': ['SiddImageDenoisingDataset'],
        'video_summarization_dataset': ['VideoSummarizationDataset'],
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
