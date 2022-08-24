# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.trainers.nlp import CsanmtTranslationTrainer
from modelscope.utils.test_utils import test_level


class TranslationTest(unittest.TestCase):
    model_id = 'damo/nlp_csanmt_translation_zh2en'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        trainer = CsanmtTranslationTrainer(model=self.model_id)
        trainer.train()


if __name__ == '__main__':
    unittest.main()
