# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class TextGenerationTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.chatglm6b_model_id = 'ZhipuAI/ChatGLM-6B'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_chatglm6b(self):
        pipe = pipeline(
            task=Tasks.text_generation,
            model=self.chatglm6b_model_id,
            model_revision='v1.0.1')
        text = '你好'
        history = []
        inputs = {'text': text, 'history': history}
        result = pipe(inputs)
        print(result)


if __name__ == '__main__':
    unittest.main()
