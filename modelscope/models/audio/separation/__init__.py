# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .mossformer import MossFormer
    from .m2.mossformer import MossFormer2

else:
    _import_structure = {
        'mossformer': ['MossFormer'],
        'm2.mossformer': ['MossFormer2'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
