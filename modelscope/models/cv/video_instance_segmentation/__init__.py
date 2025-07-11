# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .neck import MSDeformAttnPixelDecoder
    from .video_knet import KNetTrack

else:
    _import_structure = {
        'video_knet': ['KNetTrack'],
        'neck': ['MSDeformAttnPixelDecoder']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
