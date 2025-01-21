# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .backbone import (
        BertLayer,
        BertModel,
        BertPreTrainedModel,
    )
    from .configuration import BertConfig
    from .fill_mask import BertForMaskedLM
    from .text_ranking import BertForTextRanking
    from .sentence_embedding import BertForSentenceEmbedding
    from .text_classification import BertForSequenceClassification
    from .token_classification import BertForTokenClassification
    from .document_segmentation import BertForDocumentSegmentation
    from .siamese_uie import SiameseUieModel
    from .word_alignment import MBertForWordAlignment
else:
    _import_structure = {
        'backbone': [
            'BertModel',
            'BertPreTrainedModel',
        ],
        'configuration': ['BertConfig'],
        'fill_mask': ['BertForMaskedLM'],
        'text_ranking': ['BertForTextRanking'],
        'sentence_embedding': ['BertForSentenceEmbedding'],
        'text_classification': ['BertForSequenceClassification'],
        'token_classification': ['BertForTokenClassification'],
        'document_segmentation': ['BertForDocumentSegmentation'],
        'siamese_uie': ['SiameseUieModel'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
