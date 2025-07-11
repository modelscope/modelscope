# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .clip import load_clip
    from .model import SOONet
    from .tokenizer import SimpleTokenizer
    from .utils import decode_video
else:
    _import_structure = {
        'model': ['SOONet'],
        'tokenizer': ['SimpleTokenizer'],
        'utils': ['decode_video'],
        'clip': ['load_clip']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
