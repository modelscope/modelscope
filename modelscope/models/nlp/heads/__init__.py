# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .text_classification_head import TextClassificationHead
    from .torch_pretrain_head import BertMLMHead, RobertaMLMHead
else:
    _import_structure = {
        'text_classification_head': ['TextClassificationHead'],
        'torch_pretrain_head': ['BertMLMHead', 'RobertaMLMHead'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
