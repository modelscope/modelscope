# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import T5ForConditionalGeneration
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import Text2TextGenerationPipeline
from modelscope.preprocessors import Text2TextGenerationPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class Text2TextGenerationTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.model_id_generate = 'damo/t5-cn-base-test'
        self.input_generate = '中国的首都位于<extra_id_0>。'
        self.model_id_translate = 'damo/t5-translate-base-test'
        self.input_translate = 'My name is Wolfgang and I live in Berlin'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_T5(self):
        cache_path = snapshot_download(self.model_id_generate)
        model = T5ForConditionalGeneration.from_pretrained(cache_path)
        preprocessor = Text2TextGenerationPreprocessor(cache_path)
        pipeline1 = Text2TextGenerationPipeline(model, preprocessor)
        pipeline2 = pipeline(
            Tasks.text2text_generation, model=model, preprocessor=preprocessor)
        print(
            f'pipeline1: {pipeline1(self.input_generate)}\npipeline2: {pipeline2(self.input_generate)}'
        )

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_pipeline_with_model_instance(self):
        model = Model.from_pretrained(self.model_id_translate)
        preprocessor = Text2TextGenerationPreprocessor(model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.text2text_generation,
            model=model,
            preprocessor=preprocessor)
        print(pipeline_ins(self.input_translate))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_pipeline_with_model_id(self):
        pipeline_ins = pipeline(
            task=Tasks.text2text_generation, model=self.model_id_translate)
        print(pipeline_ins(self.input_translate))

    @unittest.skip(
        'only for test cases, there is no default official model yet')
    def test_run_pipeline_without_model_id(self):
        pipeline_ins = pipeline(task=Tasks.text2text_generation)
        print(pipeline_ins(self.input_generate))

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
