# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class InteractiveTranslationTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.translation
        self.model_id = 'damo/nlp_imt_translation_zh2en'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_imt_model_name_for_zh2en(self):
        inputs = '声明补充说，沃伦的同事都深感震惊，并且希望他能够投案自首。'
        prefix = 'The statement ad'
        pipeline_ins = pipeline(self.task, model=self.model_id)
        print(pipeline_ins(inputs + '<PREFIX_SPLIT>' + prefix))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_imt_model_name_for_en2zh(self):
        model_id = 'damo/nlp_imt_translation_en2zh'
        inputs = 'Elon Musk, co-founder and chief executive officer of Tesla Motors.'
        prefix = '特斯拉汽车公司'
        pipeline_ins = pipeline(self.task, model=model_id)
        print(pipeline_ins(inputs + '<PREFIX_SPLIT>' + prefix))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
