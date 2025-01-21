# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .c3d import C3D
    from .resnet2p1d import resnet26_2p1d
    from .resnet3d import resnet26_3d

else:
    _import_structure = {
        'c3d': ['C3D'],
        'resnet2p1d': ['resnet26_2p1d'],
        'resnet3d': ['resnet26_3d']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
