# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .csanmt_translation_trainer import CsanmtTranslationTrainer
    from .sentence_embedding_trainer import SentenceEmbeddingTrainer
    from .sequence_classification_trainer import SequenceClassificationTrainer
    from .siamese_uie_trainer import SiameseUIETrainer
    from .text_generation_trainer import TextGenerationTrainer
    from .text_ranking_trainer import TextRankingTrainer
    from .translation_evaluation_trainer import TranslationEvaluationTrainer
else:
    _import_structure = {
        'sequence_classification_trainer': ['SequenceClassificationTrainer'],
        'csanmt_translation_trainer': ['CsanmtTranslationTrainer'],
        'text_ranking_trainer': ['TextRankingTrainer'],
        'text_generation_trainer': ['TextGenerationTrainer'],
        'sentence_emebedding_trainer': ['SentenceEmbeddingTrainer'],
        'siamese_uie_trainer': ['SiameseUIETrainer'],
        'translation_evaluation_trainer': ['TranslationEvaluationTrainer']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
