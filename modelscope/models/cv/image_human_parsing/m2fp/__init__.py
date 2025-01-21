# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .m2fp_encoder import MSDeformAttnPixelDecoder
    from .m2fp_decoder import MultiScaleMaskedTransformerDecoder

else:
    _import_structure = {
        'm2fp_encoder': ['MSDeformAttnPixelDecoder'],
        'm2fp_decoder': ['MultiScaleMaskedTransformerDecoder'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
