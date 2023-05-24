# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .configuration import UniTEConfig
    from .translation_evaluation import UniTEForTranslationEvaluation
else:
    _import_structure = {
        'configuration': ['UniTEConfig'],
        'translation_evaluation': ['UniTEForTranslationEvaluation'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
