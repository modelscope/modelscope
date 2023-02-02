# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class TranslationTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.translation
        self.model_id = 'damo/nlp_csanmt_translation_zh2en'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_for_zh2en(self):
        inputs = '声明补充说，沃伦的同事都深感震惊，并且希望他能够投案自首。'
        pipeline_ins = pipeline(self.task, model=self.model_id)
        print(pipeline_ins(input=inputs))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_for_en2zh(self):
        model_id = 'damo/nlp_csanmt_translation_en2zh'
        inputs = 'Elon Musk, co-founder and chief executive officer of Tesla Motors.'
        pipeline_ins = pipeline(self.task, model=model_id)
        print(pipeline_ins(input=inputs))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_for_en2zh_batch(self):
        model_id = 'damo/nlp_csanmt_translation_en2zh'
        inputs = 'Elon Musk, co-founder and chief executive officer of Tesla Motors.' + \
            '<SENT_SPLIT>' + "Alibaba Group's mission is to let the world have no difficult business" + \
            '<SENT_SPLIT>' + 'Beijing is the capital of China.'
        pipeline_ins = pipeline(self.task, model=model_id)
        print(pipeline_ins(input=inputs))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_for_en2zh_base(self):
        model_id = 'damo/nlp_csanmt_translation_en2zh_base'
        inputs = 'Elon Musk, co-founder and chief executive officer of Tesla Motors.'
        pipeline_ins = pipeline(self.task, model=model_id)
        print(pipeline_ins(input=inputs))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_for_en2fr(self):
        model_id = 'damo/nlp_csanmt_translation_en2fr'
        inputs = 'When I was in my 20s, I saw my very first psychotherapy client.'
        pipeline_ins = pipeline(self.task, model=model_id)
        print(pipeline_ins(input=inputs))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_for_en2es(self):
        model_id = 'damo/nlp_csanmt_translation_en2es'
        inputs = 'When I was in my 20s, I saw my very first psychotherapy client.'
        pipeline_ins = pipeline(self.task, model=model_id)
        print(pipeline_ins(input=inputs))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_for_fr2en(self):
        model_id = 'damo/nlp_csanmt_translation_fr2en'
        inputs = "Quand j'avais la vingtaine, j'ai vu mes tout premiers clients comme psychothérapeute."
        pipeline_ins = pipeline(self.task, model=model_id)
        print(pipeline_ins(input=inputs))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_for_es2en(self):
        model_id = 'damo/nlp_csanmt_translation_es2en'
        inputs = 'Los físicos clasifican las partículas en dos categorías.'
        pipeline_ins = pipeline(self.task, model=model_id)
        print(pipeline_ins(input=inputs))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        inputs = '声明补充说，沃伦的同事都深感震惊，并且希望他能够投案自首。'
        pipeline_ins = pipeline(self.task)
        print(pipeline_ins(input=inputs))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
