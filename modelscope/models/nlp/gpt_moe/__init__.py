# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .configuration import GPTMoEConfig
    from .backbone import GPTMoEModel
    from .text_generation import GPTMoEForTextGeneration
    from .tokenizer import JiebaBPETokenizer
    from .distributed_gpt_moe import DistributedGPTMoE
else:
    _import_structure = {
        'configuration': ['GPTMoEConfig'],
        'backbone': ['GPTMoEModel'],
        'text_generation': ['GPTMoEForTextGeneration'],
        'tokenizer': ['JiebaBPETokenizer'],
        'distributed_gpt_moe': ['DistributedGPTMoE'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
