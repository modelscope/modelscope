# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .clip import CLIPTrainer
    from .team import TEAMImgClsTrainer
    from .ofa import OFATrainer
    from .mplug import MPlugTrainer

else:
    _import_structure = {
        'clip': ['CLIPTrainer'],
        'team': ['TEAMImgClsTrainer'],
        'ofa': ['OFATrainer'],
        'mplug': ['MPlugTrainer'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
