# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .image_color_enhance import ImageColorEnhance
    from .adaint import AdaIntImageColorEnhance
    from .deeplpf import DeepLPFImageColorEnhance

else:
    _import_structure = {
        'image_color_enhance': ['ImageColorEnhance'],
        'adaint': ['AdaIntImageColorEnhance'],
        'deeplpf': ['DeepLPFImageColorEnhance']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
