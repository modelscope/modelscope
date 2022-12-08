# Modified by Zhipu.AI
# Original Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .mglm_for_text_summarization import MGLMForTextSummarization
else:
    _import_structure = {
        'mglm_for_text_summarization': ['MGLMForTextSummarization'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
