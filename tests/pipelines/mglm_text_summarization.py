# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.preprocessors import MGLMSummarizationPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class mGLMTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.output_dir = 'unittest_output'
        os.makedirs(self.output_dir, exist_ok=True)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_mglm_with_name(self):
        model = 'ZhipuAI/Multilingual-GLM-Summarization-zh'
        preprocessor = MGLMSummarizationPreprocessor()
        pipe = pipeline(
            task=Tasks.summarization,
            model=model,
            preprocessor=preprocessor,
        )
        result = pipe(
            '二十大代表、中国航天科技集团有限公司党组书记、董事长吴燕生介绍：进入新时代，中国航天取得了历史性的成就。我在这里再以运载火箭的发射为例。党的十八大以来的十年，中国航天进行了274次发射。党的十九大以来的五年，我们进行了192次火箭发射，占长征系列火箭发射总数的43.2%。正是这些成就推动我国全面建成了航天大国，进入了航天强国的行列。'  # noqa
        )
        print(result)


if __name__ == '__main__':
    unittest.main()
