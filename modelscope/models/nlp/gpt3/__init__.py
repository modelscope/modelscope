# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .configuration_gpt3 import GPT3Config
    from .modeling_gpt3 import GPT3Model
    from .gpt3_for_text_generation import GPT3ForTextGeneration
    from .tokenizer_gpt3 import JiebaBPETokenizer
else:
    _import_structure = {
        'configuration_gpt3': ['GPT3Config'],
        'modeling_gpt3': ['GPT3Model'],
        'gpt3_for_text_generation': ['GPT3ForTextGeneration'],
        'tokenizer_gpt3': ['JiebaBPETokenizer'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
