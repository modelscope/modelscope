# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .maskdino_encoder import MaskDINOEncoder
    from .maskdino_decoder import MaskDINODecoder

else:
    _import_structure = {
        'maskdino_encoder': ['MaskDINOEncoder'],
        'maskdino_decoder': ['MaskDINODecoder'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
