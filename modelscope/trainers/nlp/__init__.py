# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .sequence_classification_trainer import SequenceClassificationTrainer
    from .csanmt_translation_trainer import CsanmtTranslationTrainer
    from .passage_ranking_trainer import PassageRankingTranier
else:
    _import_structure = {
        'sequence_classification_trainer': ['SequenceClassificationTrainer'],
        'csanmt_translation_trainer': ['CsanmtTranslationTrainer'],
        'passage_ranking_trainer': ['PassageRankingTrainer']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
