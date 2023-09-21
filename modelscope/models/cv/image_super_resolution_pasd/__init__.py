# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .unet_2d_condition import UNet2DConditionModel
    from .controlnet import ControlNetModel

else:
    _import_structure = {
        'unet_2d_condition': ['UNet2DConditionModel'],
        'controlnet': ['ControlNetModel']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
