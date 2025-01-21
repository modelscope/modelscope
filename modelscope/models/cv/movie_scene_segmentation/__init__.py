# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:

    from .model import MovieSceneSegmentationModel
    from .datasets import MovieSceneSegmentationDataset

else:
    _import_structure = {
        'model': ['MovieSceneSegmentationModel'],
        'datasets': ['MovieSceneSegmentationDataset'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
