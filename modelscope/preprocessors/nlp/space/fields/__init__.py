# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .gen_field import MultiWOZBPETextField
    from .intent_field import IntentBPETextField
else:
    _import_structure = {
        'gen_field': ['MultiWOZBPETextField'],
        'intent_field': ['IntentBPETextField']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
