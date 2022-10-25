# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .conversational_text_to_sql_preprocessor import \
        ConversationalTextToSqlPreprocessor
    from .fields import (get_label, SubPreprocessor, preprocess_dataset,
                         process_dataset)

else:
    _import_structure = {
        'conversational_text_to_sql_preprocessor':
        ['ConversationalTextToSqlPreprocessor'],
        'fields': [
            'get_label', 'SubPreprocessor', 'preprocess_dataset',
            'process_dataset'
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
