# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.models.nlp.unite.configuration_unite import EvaluationMode
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class TranslationEvaluationTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.translation_evaluation
        self.model_id_large = 'damo/nlp_unite_mup_translation_evaluation_multilingual_large'
        self.model_id_base = 'damo/nlp_unite_mup_translation_evaluation_multilingual_base'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_for_unite_large(self):
        input = {
            'hyp': [
                'This is a sentence.',
                'This is another sentence.',
            ],
            'src': [
                '这是个句子。',
                '这是另一个句子。',
            ],
            'ref': [
                'It is a sentence.',
                'It is another sentence.',
            ]
        }

        pipeline_ins = pipeline(self.task, model=self.model_id_large)
        print(pipeline_ins(input=input))

        pipeline_ins.change_eval_mode(eval_mode=EvaluationMode.SRC)
        print(pipeline_ins(input=input))

        pipeline_ins.change_eval_mode(eval_mode=EvaluationMode.REF)
        print(pipeline_ins(input=input))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_for_unite_base(self):
        input = {
            'hyp': [
                'This is a sentence.',
                'This is another sentence.',
            ],
            'src': [
                '这是个句子。',
                '这是另一个句子。',
            ],
            'ref': [
                'It is a sentence.',
                'It is another sentence.',
            ]
        }

        pipeline_ins = pipeline(self.task, model=self.model_id_base)
        print(pipeline_ins(input=input))

        pipeline_ins.change_eval_mode(eval_mode=EvaluationMode.SRC)
        print(pipeline_ins(input=input))

        pipeline_ins.change_eval_mode(eval_mode=EvaluationMode.REF)
        print(pipeline_ins(input=input))


if __name__ == '__main__':
    unittest.main()
