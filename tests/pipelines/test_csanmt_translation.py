# Copyright (c) Alibaba, Inc. and its affiliates.
import shutil
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import TranslationPipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class TranslationTest(unittest.TestCase):
    model_id = 'damo/nlp_csanmt_translation_zh2en'
    inputs = '声明 补充 说 ， 沃伦 的 同事 都 深感 震惊 ， 并且 希望 他 能够 投@@ 案@@ 自@@ 首 。'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(task=Tasks.translation, model=self.model_id)
        print(pipeline_ins(input=self.inputs))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.translation)
        print(pipeline_ins(input=self.inputs))


if __name__ == '__main__':
    unittest.main()
