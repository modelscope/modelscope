# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .optimizers.child_tuning_adamw_optimizer import calculate_fisher, ChildTuningAdamW
    from .adapter import Adapter, AdapterConfig, AdapterModule
    from .lora import LoRA, LoRAConfig, Linear, MergedLinear, Embedding, Conv2d
    from .prompt import Prompt, PromptConfig, PromptModule
    from .control_sd_lora import ControlLoRACrossAttnProcessor, ControlLoRACrossAttnProcessorV2, ControlLoRATuner
    from .base import SwiftConfig, Swift
else:
    _import_structure = {
        'optimizers.child_tuning_adamw_optimizer':
        ['calculate_fisher', 'ChildTuningAdamW'],
        'adapter': ['Adapter', 'AdapterConfig', 'AdapterModule'],
        'lora': [
            'LoRA', 'LoRAConfig', 'Linear', 'MergedLinear', 'Embedding',
            'Conv2d'
        ],
        'prompt': ['Prompt', 'PromptConfig', 'PromptModule'],
        'control_sd_lora': [
            'ControlLoRACrossAttnProcessor', 'ControlLoRACrossAttnProcessorV2',
            'ControlLoRATuner'
        ],
        'base': ['SwiftConfig', 'Swift']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
