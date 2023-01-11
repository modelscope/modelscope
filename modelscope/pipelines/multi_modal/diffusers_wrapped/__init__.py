# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .stable_diffusion import StableDiffusionWrapperPipeline
    from .stable_diffusion import ChineseStableDiffusionPipeline
else:
    _import_structure = {
        'stable_diffusion':
        ['StableDiffusionWrapperPipeline', 'ChineseStableDiffusionPipeline']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
