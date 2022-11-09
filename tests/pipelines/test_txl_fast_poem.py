# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.preprocessors import TXLFastPoemPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class TXLTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.output_dir = 'unittest_output'
        os.makedirs(self.output_dir, exist_ok=True)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_TXL_with_name(self):
        model = 'ZhipuAI/TransformerXL-Fast-Poem'
        preprocessor = TXLFastPoemPreprocessor()
        pipe = pipeline(
            task=Tasks.fast_poem,
            model=model,
            preprocessor=preprocessor,
        )
        inputs = {
            'title': '明月',
            'author': '杜甫',
            'desc': '寂寞',
            'lycr': 7,
            'senlength': 4
        }
        result = pipe(inputs)
        print(result)


if __name__ == '__main__':
    unittest.main()
