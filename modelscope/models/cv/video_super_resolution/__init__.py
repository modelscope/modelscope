# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .real_basicvsr_for_video_super_resolution import RealBasicVSRNetForVideoSR

else:
    _import_structure = {
        'real_basicvsr_for_video_super_resolution':
        ['RealBasicVSRNetForVideoSR']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
