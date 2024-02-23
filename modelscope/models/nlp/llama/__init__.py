# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from transformers import LlamaTokenizer
from transformers.models.llama import LlamaConfig, LlamaTokenizerFast

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .backbone import LlamaModel
    from .text_generation import LlamaForTextGeneration
else:
    _import_structure = {
        'backbone': ['LlamaModel'],
        'text_generation': ['LlamaForTextGeneration'],
    }
    _extra_objects = {
        'LlamaConfig': LlamaConfig,
        'LlamaTokenizer': LlamaTokenizer,
        'LlamaTokenizerFast': LlamaTokenizerFast,
    }
    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects=_extra_objects,
    )
