# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:

    from .model import create_model, load_model_wo_clip
    from .modules.cfg_sampler import ClassifierFreeSampleModel
else:
    _import_structure = {
        'model': ['create_model', 'load_model_wo_clip'],
        'modules.cfg_sampler': ['ClassifierFreeSampleModel']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
