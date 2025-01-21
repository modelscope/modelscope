# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .csanmt_for_translation_exporter import CsanmtForTranslationExporter
    from .model_for_token_classification_exporter import \
        ModelForSequenceClassificationExporter
    from .sbert_for_sequence_classification_exporter import \
        SbertForSequenceClassificationExporter
    from .sbert_for_zero_shot_classification_exporter import \
        SbertForZeroShotClassificationExporter
else:
    _import_structure = {
        'csanmt_for_translation_exporter': ['CsanmtForTranslationExporter'],
        'model_for_token_classification_exporter':
        ['ModelForSequenceClassificationExporter'],
        'sbert_for_zero_shot_classification_exporter':
        ['SbertForZeroShotClassificationExporter'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
