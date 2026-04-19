# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.import_utils import exists


class LanguageIdentificationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.text_classification
        self.model_id = 'damo/nlp_language_identification-classification-base'

    @unittest.skipUnless(
        exists('tensorflow'), 'Skip because tensorflow is not installed.')
    def test_run_with_model_name_for_en2de(self):
        inputs = 'Elon Musk, co-founder and chief executive officer of Tesla Motors.\n' \
                 'Gleichzeitig nahm die Legion an der Befriedung Algeriens teil, die von.\n' \
                 '使用pipeline推理及在线体验功能的时候，尽量输入单句文本，如果是多句长文本建议人工分句。'
        pipeline_ins = pipeline(self.task, model=self.model_id)
        print(pipeline_ins(input=inputs))


if __name__ == '__main__':
    unittest.main()
