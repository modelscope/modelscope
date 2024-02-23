# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .marigold import MarigoldDepthOutput
    from .marigold_utils import (chw2hwc, colorize_depth_maps, ensemble_depths,
                                 find_batch_size, inter_distances,
                                 resize_max_res)
else:
    _import_structure = {
        'marigold': ['MarigoldDepthOutput'],
        'marigold_utils': [
            'find_batch_size', 'inter_distances', 'ensemble_depths',
            'colorize_depth_maps', 'chw2hwc', 'resize_max_res'
        ]
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
