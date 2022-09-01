# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class TranslationTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_for_zh2en(self):
        model_id = 'damo/nlp_csanmt_translation_zh2en'
        inputs = '声明补充说，沃伦的同事都深感震惊，并且希望他能够投案自首。'
        pipeline_ins = pipeline(task=Tasks.translation, model=model_id)
        print(pipeline_ins(input=inputs))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_for_en2zh(self):
        model_id = 'damo/nlp_csanmt_translation_en2zh'
        inputs = 'Elon Musk, co-founder and chief executive officer of Tesla Motors.'
        pipeline_ins = pipeline(task=Tasks.translation, model=model_id)
        print(pipeline_ins(input=inputs))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        inputs = '声明补充说，沃伦的同事都深感震惊，并且希望他能够投案自首。'
        pipeline_ins = pipeline(task=Tasks.translation)
        print(pipeline_ins(input=inputs))


if __name__ == '__main__':
    unittest.main()
