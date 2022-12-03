# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.trainers.nlp import CsanmtTranslationTrainer
from modelscope.utils.test_utils import test_level


class TranslationTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_for_en2zh(self):
        model_id = 'damo/nlp_csanmt_translation_en2zh'
        trainer = CsanmtTranslationTrainer(model=model_id)
        trainer.train()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_for_en2fr(self):
        model_id = 'damo/nlp_csanmt_translation_en2fr'
        trainer = CsanmtTranslationTrainer(model=model_id)
        trainer.train()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_for_en2es(self):
        model_id = 'damo/nlp_csanmt_translation_en2es'
        trainer = CsanmtTranslationTrainer(model=model_id)
        trainer.train()


if __name__ == '__main__':
    unittest.main()
