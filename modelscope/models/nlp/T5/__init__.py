# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .backbone import T5Model
    from .text2text_generation import T5ForConditionalGeneration

else:
    _import_structure = {
        'backbone': ['T5Model'],
        'text2text_generation': ['T5ForConditionalGeneration'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
