# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .configuration import PlugNLGConfig
    from .backbone import PlugModel
    from .distributed_plug import DistributedPlug
else:
    _import_structure = {
        'configuration': ['PlugNLGConfig'],
        'backbone': ['PlugModel'],
        'distributed_plug': ['DistributedPlug'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
