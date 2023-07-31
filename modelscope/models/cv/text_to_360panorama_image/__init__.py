# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .pipeline_base import StableDiffusionBlendExtendPipeline
    from .pipeline_sr import StableDiffusionControlNetImg2ImgPanoPipeline

else:
    _import_structure = {
        'pipeline_base': ['StableDiffusionBlendExtendPipeline'],
        'pipeline_sr': ['StableDiffusionControlNetImg2ImgPanoPipeline'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
