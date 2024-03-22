# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .backbone import BloomModel
    from .text_generation import BloomForTextGeneration
    from .sentence_embedding import BloomForSentenceEmbedding
else:
    _import_structure = {
        'backbone': ['BloomModel'],
        'text_generation': ['BloomForTextGeneration'],
        'sentence_embedding': ['BloomForSentenceEmbedding']
    }
    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
