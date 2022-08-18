# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .configuration_plug import PlugNLGConfig
    from .modeling_plug import PlugModel
    from .distributed_plug import DistributedPlug
    from .plug_for_text_generation import PlugForTextGeneration
else:
    _import_structure = {
        'configuration_plug': ['PlugNLGConfig'],
        'modeling_plug': ['PlugModel'],
        'distributed_plug': ['DistributedPlug'],
        'plug_for_text_generation': ['PlugForTextGeneration'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
