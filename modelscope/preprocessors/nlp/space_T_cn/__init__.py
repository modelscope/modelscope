# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .table_question_answering_preprocessor import TableQuestionAnsweringPreprocessor
    from .fields import MultiWOZBPETextField, IntentBPETextField

else:
    _import_structure = {
        'table_question_answering_preprocessor':
        ['TableQuestionAnsweringPreprocessor'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
