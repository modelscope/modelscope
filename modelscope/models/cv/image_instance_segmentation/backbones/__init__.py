# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .resnet import build_resnet_backbone
    from .swin_transformer import D2SwinTransformer, SwinTransformer

else:
    _import_structure = {
        'swin_transformer': ['SwinTransformer', 'D2SwinTransformer'],
        'resnet': ['build_resnet_backbone']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
