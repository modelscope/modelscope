# Copyright (c) Alibaba, Inc. and its affiliates.
import shutil
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class TranslationTest(unittest.TestCase):
    model_id = 'damo/nlp_csanmt_translation'
    inputs = 'Gut@@ ach : Incre@@ ased safety for pedestri@@ ans'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(task=Tasks.translation, model=self.model_id)
        print(pipeline_ins(input=self.inputs))


if __name__ == '__main__':
    unittest.main()
