# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path
import shutil
import unittest

from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.utils.hub import read_config
from modelscope.utils.test_utils import test_level


class TranslationEvaluationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.name = Trainers.translation_evaluation_trainer
        self.model_id_large = 'damo/nlp_unite_mup_translation_evaluation_multilingual_large'
        self.model_id_base = 'damo/nlp_unite_mup_translation_evaluation_multilingual_base'

    def tearDown(self) -> None:
        cfg_base = read_config(self.model_id_base)
        if os.path.exists(cfg_base.train.work_dir):
            shutil.rmtree(cfg_base.train.work_dir, ignore_errors=True)
        cfg_large = read_config(self.model_id_large)
        if os.path.exists(cfg_large.train.work_dir):
            shutil.rmtree(cfg_large.train.work_dir, ignore_errors=True)
        super().tearDown()

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
