# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .nerf_recon_4k import NeRFRecon4K
    from .nerf_preprocess import NeRFReconPreprocessor

else:
    _import_structure = {'nerf_recon_4k': ['NeRFRecon4K']}
    _import_structure = {'nerf_preprocess': ['NeRFReconPreprocessor']}

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
