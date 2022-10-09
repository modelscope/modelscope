# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import numpy as np

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import FeatureExtractionModel
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import FeatureExtractionPipeline
from modelscope.preprocessors import NLPPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class FeatureExtractionTaskModelTest(unittest.TestCase,
                                     DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.feature_extraction
        self.model_id = 'damo/pert_feature-extraction_base-test'

    sentence1 = '测试embedding'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_direct_file_download(self):
        cache_path = snapshot_download(self.model_id)
        tokenizer = NLPPreprocessor(cache_path, padding=False)
        model = FeatureExtractionModel.from_pretrained(self.model_id)
        pipeline1 = FeatureExtractionPipeline(model, preprocessor=tokenizer)
        pipeline2 = pipeline(
            Tasks.feature_extraction, model=model, preprocessor=tokenizer)
        result = pipeline1(input=self.sentence1)

        print(f'sentence1: {self.sentence1}\n'
              f'pipeline1:{np.shape(result[OutputKeys.TEXT_EMBEDDING])}')
        result = pipeline2(input=self.sentence1)
        print(f'sentence1: {self.sentence1}\n'
              f'pipeline1: {np.shape(result[OutputKeys.TEXT_EMBEDDING])}')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        tokenizer = NLPPreprocessor(model.model_dir, padding=False)
        pipeline_ins = pipeline(
            task=Tasks.feature_extraction, model=model, preprocessor=tokenizer)
        result = pipeline_ins(input=self.sentence1)
        print(np.shape(result[OutputKeys.TEXT_EMBEDDING]))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.feature_extraction, model=self.model_id)
        result = pipeline_ins(input=self.sentence1)
        print(np.shape(result[OutputKeys.TEXT_EMBEDDING]))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.feature_extraction)
        result = pipeline_ins(input=self.sentence1)
        print(np.shape(result[OutputKeys.TEXT_EMBEDDING]))


if __name__ == '__main__':
    unittest.main()
