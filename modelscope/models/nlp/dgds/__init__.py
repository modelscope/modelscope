# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .document_grounded_dialog_generate import DocumentGroundedDialogGenerateModel
    from .document_grounded_dialog_retrieval import DocumentGroundedDialogRerankModel
    from .document_grounded_dialog_retrieval import DocumentGroundedDialogRetrievalModel
else:
    _import_structure = {
        'document_grounded_dialog_generate':
        ['DocumentGroundedDialogGenerateModel'],
        'document_grounded_dialog_rerank':
        ['DocumentGroundedDialogRerankModel'],
        'document_grounded_dialog_retrieval':
        ['DocumentGroundedDialogRetrievalModel']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
