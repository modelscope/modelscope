# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.preprocessors import CodeGeeXPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class CodeGeeXCodeTranslationTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.output_dir = 'unittest_output'
        os.makedirs(self.output_dir, exist_ok=True)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_CodeGeeX_with_name(self):
        model = 'ZhipuAI/CodeGeeX-Code-Translation-13B'
        preprocessor = CodeGeeXPreprocessor()
        pipe = pipeline(
            task=Tasks.code_translation,
            model=model,
            preprocessor=preprocessor,
        )
        inputs = {
            'prompt': 'for i in range(10):\n\tprint(i)\n',
            'source language': 'Python',
            'target language': 'C++'
        }
        result = pipe(inputs)
        print(result)


if __name__ == '__main__':
    unittest.main()
