# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .human_wholebody_keypoint_dataset import WholeBodyCocoTopDownDataset

else:
    _import_structure = {
        'human_wholebody_keypoint_dataset': ['WholeBodyCocoTopDownDataset']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
