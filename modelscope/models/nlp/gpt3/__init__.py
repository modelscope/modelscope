# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .configuration import GPT3Config
    from .backbone import GPT3Model
    from .text_generation import GPT3ForTextGeneration
    from .tokenizer import JiebaBPETokenizer
    from .distributed_gpt3 import DistributedGPT3
else:
    _import_structure = {
        'configuration': ['GPT3Config'],
        'backbone': ['GPT3Model'],
        'text_generation': ['GPT3ForTextGeneration'],
        'tokenizer': ['JiebaBPETokenizer'],
        'distributed_gpt3': ['DistributedGPT3'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
