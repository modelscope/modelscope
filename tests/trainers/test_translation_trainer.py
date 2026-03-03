# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.utils.import_utils import exists


class TranslationTest(unittest.TestCase):

    @unittest.skipUnless(
        exists('tensorflow'), 'Skip test because tensorflow is not installed.')
    def test_run_with_model_name_for_en2zh(self):
        from modelscope.trainers.nlp import CsanmtTranslationTrainer
        model_id = 'damo/nlp_csanmt_translation_en2zh'
        trainer = CsanmtTranslationTrainer(model=model_id)
        trainer.train()

    @unittest.skipUnless(
        exists('tensorflow'), 'Skip test because tensorflow is not installed.')
    def test_run_with_model_name_for_en2fr(self):
        from modelscope.trainers.nlp import CsanmtTranslationTrainer
        model_id = 'damo/nlp_csanmt_translation_en2fr'
        trainer = CsanmtTranslationTrainer(model=model_id)
        trainer.train()

    @unittest.skipUnless(
        exists('tensorflow'), 'Skip test because tensorflow is not installed.')
    def test_run_with_model_name_for_en2es(self):
        from modelscope.trainers.nlp import CsanmtTranslationTrainer
        model_id = 'damo/nlp_csanmt_translation_en2es'
        trainer = CsanmtTranslationTrainer(model=model_id)
        trainer.train()


if __name__ == '__main__':
    unittest.main()
