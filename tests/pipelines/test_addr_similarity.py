# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import SbertForSequenceClassification
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import TextClassificationPipeline
from modelscope.preprocessors import TextClassificationTransformersPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.regress_test_utils import IgnoreKeyFn, MsRegressTool
from modelscope.utils.test_utils import test_level


class AddrSimilarityTest(unittest.TestCase, DemoCompatibilityCheck):

    sentence1 = '阿里巴巴西溪园区'
    sentence2 = '文一西路阿里巴巴'
    model_id = 'damo/nlp_structbert_address-matching_chinese_base'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        preprocessor = TextClassificationTransformersPreprocessor(
            model.model_dir)

        pipeline_ins = pipeline(
            task=Tasks.text_classification,
            model=model,
            preprocessor=preprocessor)
        print(pipeline_ins(input=(self.sentence1, self.sentence2)))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.text_classification, model=self.model_id)
        print(pipeline_ins(input=(self.sentence1, self.sentence2)))

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
