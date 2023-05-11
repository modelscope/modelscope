# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.models.nlp.unite.configuration import InputFormat
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
        input_dict = {
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
        print(pipeline_ins(input_dict)['score'])

        pipeline_ins.change_input_format(input_format=InputFormat.SRC)
        print(pipeline_ins(input_dict)['score'])

        pipeline_ins.change_input_format(input_format=InputFormat.REF)
        print(pipeline_ins(input_dict)['score'])

        pipeline_ins = pipeline(
            self.task, model=self.model_id_large, device='cpu')
        print(pipeline_ins(input_dict)['score'])

        pipeline_ins.change_input_format(input_format=InputFormat.SRC)
        print(pipeline_ins(input_dict)['score'])

        pipeline_ins.change_input_format(input_format=InputFormat.REF)
        print(pipeline_ins(input_dict)['score'])

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_for_unite_base(self):
        input_dict = {
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
        print(pipeline_ins(input_dict)['score'])

        pipeline_ins.change_input_format(input_format=InputFormat.SRC)
        print(pipeline_ins(input_dict)['score'])

        pipeline_ins.change_input_format(input_format=InputFormat.REF)
        print(pipeline_ins(input_dict)['score'])

        pipeline_ins = pipeline(
            self.task, model=self.model_id_base, device='cpu')
        print(pipeline_ins(input_dict)['score'])

        pipeline_ins.change_input_format(input_format=InputFormat.SRC)
        print(pipeline_ins(input_dict)['score'])

        pipeline_ins.change_input_format(input_format=InputFormat.REF)
        print(pipeline_ins(input_dict)['score'])


if __name__ == '__main__':
    unittest.main()
