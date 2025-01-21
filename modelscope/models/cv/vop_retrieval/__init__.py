# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .basic_utils import set_seed, get_state_dict, load_data, init_transform_dict, load_frames_from_video
    from .model import VoP
    from .model_se import VoP_SE
    from .tokenization_clip import LengthAdaptiveTokenizer
else:
    _import_structure = {
        'basic_utils': [
            'set_seed', 'get_state_dict', 'load_data', 'init_transform_dict',
            'load_frames_from_video'
        ],
        'model': ['VoP'],
        'model_se': ['VideoTextRetrievalModelSeries'],
        'tokenization_clip': ['LengthAdaptiveTokenizer']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
