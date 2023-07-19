# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .configuration import Llama2Config
    from .text_generation import Llama2ForTextGeneration
    from .backbone import Llama2Model
    from .tokenization import Llama2Tokenizer
    from .tokenization_fast import Llama2TokenizerFast
else:
    _import_structure = {
        'configuration': ['Llama2Config'],
        'text_generation': ['Llama2ForTextGeneration'],
        'backbone': ['Llama2Model'],
        'tokenization': ['Llama2Tokenizer'],
        'tokenization_fast': ['Llama2TokenizerFast'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
