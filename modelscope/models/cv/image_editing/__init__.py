# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .masactrl import MutualSelfAttentionControl
    from .masactrl_utils import register_attention_editor_diffusers
else:
    _import_structure = {
        'masactrl': ['MutualSelfAttentionControl'],
        'masactrl_utils': ['register_attention_editor_diffusers']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
