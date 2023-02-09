# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:

    from .vision_efficient_tuning_adapter import VisionEfficientTuningAdapterModel
    from .vision_efficient_tuning_prompt import VisionEfficientTuningPromptModel
    from .vision_efficient_tuning_prefix import VisionEfficientTuningPrefixModel
    from .vision_efficient_tuning_lora import VisionEfficientTuningLoRAModel

else:
    _import_structure = {
        'vision_efficient_tuning_adapter':
        ['VisionEfficientTuningAdapterModel'],
        'vision_efficient_tuning_prompt': ['VisionEfficientTuningPromptModel'],
        'vision_efficient_tuning_prefix': ['VisionEfficientTuningPrefixModel'],
        'vision_efficient_tuning_lora': ['VisionEfficientTuningLoRAModel'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
