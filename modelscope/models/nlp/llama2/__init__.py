# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.models.nlp.llama import LlamaConfig as Llama2Config
from modelscope.models.nlp.llama import LlamaTokenizer as Llama2Tokenizer
from modelscope.models.nlp.llama import \
    LlamaTokenizerFast as Llama2TokenizerFast
from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .backbone import Llama2Model
    from .text_generation import Llama2ForTextGeneration
else:
    _import_structure = {
        'backbone': ['Llama2Model'],
        'text_generation': ['Llama2ForTextGeneration'],
    }
    _extra_objects = {
        'Llama2Config': Llama2Config,
        'Llama2Tokenizer': Llama2Tokenizer,
        'Llama2TokenizerFast': Llama2TokenizerFast,
    }
    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects=_extra_objects,
    )
