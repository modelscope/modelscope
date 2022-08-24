# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .sequence_classification_head import SequenceClassificationHead
    from .torch_pretrain_head import BertMLMHead, RobertaMLMHead
else:
    _import_structure = {
        'sequence_classification_head': ['SequenceClassificationHead'],
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
