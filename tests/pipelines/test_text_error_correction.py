# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import BartForTextErrorCorrection
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import TextErrorCorrectionPipeline
from modelscope.preprocessors import TextErrorCorrectionPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class TextErrorCorrectionTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.text_error_correction
        self.model_id = 'damo/nlp_bart_text-error-correction_chinese'

    input = '随着中国经济突飞猛近，建造工业与日俱增'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_direct_download(self):
        cache_path = snapshot_download(self.model_id)
        model = BartForTextErrorCorrection(cache_path)
        preprocessor = TextErrorCorrectionPreprocessor(cache_path)
        pipeline1 = TextErrorCorrectionPipeline(model, preprocessor)
        pipeline2 = pipeline(
            Tasks.text_error_correction,
            model=model,
            preprocessor=preprocessor)
        print(
            f'pipeline1: {pipeline1(self.input)}\npipeline2: {pipeline2(self.input)}'
        )

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        preprocessor = TextErrorCorrectionPreprocessor(model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.text_error_correction,
            model=model,
            preprocessor=preprocessor)
        print(pipeline_ins(self.input))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.text_error_correction, model=self.model_id)
        print(pipeline_ins(self.input))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.text_error_correction)
        print(pipeline_ins(self.input))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
