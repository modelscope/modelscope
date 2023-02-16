# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .configuration import MegatronBertConfig
    from .backbone import MegatronBertModel
    from .fill_mask import MegatronBertForMaskedLM
else:
    _import_structure = {
        'configuration': ['MegatronBertConfig'],
        'backbone': ['MegatronBertModel'],
        'fill_mask': ['MegatronBertForMaskedLM'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
