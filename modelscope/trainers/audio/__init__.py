# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    print('TYPE_CHECKING...')
    from .ans_trainer import ANSTrainer
    from .kws_farfield_trainer import KWSFarfieldTrainer
    from .kws_nearfield_trainer import KWSNearfieldTrainer
    from .tts_trainer import KanttsTrainer

else:
    _import_structure = {
        'tts_trainer': ['KanttsTrainer'],
        'ans_trainer': ['ANSTrainer'],
        'kws_nearfield_trainer': ['KWSNearfieldTrainer'],
        'kws_farfield_trainer': ['KWSFarfieldTrainer'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
