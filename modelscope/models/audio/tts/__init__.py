# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .sambert_hifi import SambertHifigan
    from .laura_codec import LauraCodecGenModel

else:
    _import_structure = {
        'sambert_hifi': ['SambertHifigan'],
        'laura_codec': ['LauraCodecGenModel'],
    }
    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
