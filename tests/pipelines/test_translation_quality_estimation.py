# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class TranslationQualityEstimationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.sentence_similarity
        self.model_id = 'damo/nlp_translation_quality_estimation_multilingual'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_for_en2zh(self):
        inputs = {
            'source_text': 'Love is a losing game',
            'target_text': '宝贝，人和人一场游戏'
        }
        pipeline_ins = pipeline(self.task, model=self.model_id)
        print(pipeline_ins(input=inputs))


if __name__ == '__main__':
    unittest.main()
