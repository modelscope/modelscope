# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.utils.test_utils import test_level


class TranslationEvaluationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.name = Trainers.translation_evaluation_trainer
        self.model_id_large = 'damo/nlp_unite_mup_translation_evaluation_multilingual_large'
        self.model_id_base = 'damo/nlp_unite_mup_translation_evaluation_multilingual_base'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_unite_mup_large(self) -> None:
        default_args = {'model': self.model_id_large}
        trainer = build_trainer(name=self.name, default_args=default_args)
        trainer.train()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_unite_mup_base(self) -> None:
        default_args = {'model': self.model_id_base}
        trainer = build_trainer(name=self.name, default_args=default_args)
        trainer.train()


if __name__ == '__main__':
    unittest.main()
