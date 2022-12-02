# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .configuration_unite import UniTEConfig
    from .modeling_unite import UniTEForTranslationEvaluation
else:
    _import_structure = {
        'configuration_unite': ['UniTEConfig'],
        'modeling_unite': ['UniTEForTranslationEvaluation'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
