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
            task=Tasks.text_summarization,
            model=model,
            preprocessor=preprocessor,
        )
        result = pipe(
            '据中国载人航天工程办公室消息，北京时间2022年10月25日，梦天实验舱与长征五号B遥四运载火箭组合体已转运至发射区。后续将按计划开展发射前各项功能检查和联合测试等工作，计划于近日择机实施发射。目前，文昌航天发射场设施设备状态良好，参试各单位正在加紧开展任务准备，全力以赴确保空间站建造任务决战决胜。'  # noqa
        )
        print(result)

        model = 'ZhipuAI/Multilingual-GLM-Summarization-en'
        preprocessor = MGLMSummarizationPreprocessor()
        pipe = pipeline(
            task=Tasks.summarization,
            model=model,
            preprocessor=preprocessor,
        )
        result = pipe(
            '据中国载人航天工程办公室消息，北京时间2022年10月25日，梦天实验舱与长征五号B遥四运载火箭组合体已转运至发射区。后续将按计划开展发射前各项功能检查和联合测试等工作，计划于近日择机实施发射。目前，文昌航天发射场设施设备状态良好，参试各单位正在加紧开展任务准备，全力以赴确保空间站建造任务决战决胜。'  # noqa
        )
        print(result)


if __name__ == '__main__':
    unittest.main()
