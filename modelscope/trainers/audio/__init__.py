# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    print('TYPE_CHECKING...')
    from .tts_trainer import KanttsTrainer
    from .ans_trainer import ANSTrainer

else:
    _import_structure = {
        'tts_trainer': ['KanttsTrainer'],
        'ans_trainer': ['ANSTrainer']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
