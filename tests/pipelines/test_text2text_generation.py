# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import T5ForConditionalGeneration
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import TextGenerationT5Pipeline
from modelscope.preprocessors import TextGenerationT5Preprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class Text2TextGenerationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id_generate = 'damo/t5-cn-base-test'
        self.input_generate = '中国的首都位于<extra_id_0>。'
        self.model_id_translate = 'damo/t5-translate-base-test'
        self.input_translate = 'My name is Wolfgang and I live in Berlin'
        self.model_id_rewriting = 'damo/nlp_mt5_dialogue-rewriting_chinese-base'
        self.input_rewriting = '杨阳胖吗[SEP]我一个同学叫杨阳[SEP]他多少斤'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_T5(self):
        cache_path = snapshot_download(self.model_id_generate)
        model = T5ForConditionalGeneration.from_pretrained(cache_path)
        preprocessor = TextGenerationT5Preprocessor(cache_path)
        pipeline1 = TextGenerationT5Pipeline(model, preprocessor)
        pipeline2 = pipeline(
            Tasks.text2text_generation, model=model, preprocessor=preprocessor)
        print(
            f'pipeline1: {pipeline1(self.input_generate)}\npipeline2: {pipeline2(self.input_generate)}'
        )

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_pipeline_with_model_instance(self):
        model = Model.from_pretrained(self.model_id_translate)
        preprocessor = TextGenerationT5Preprocessor(model.model_dir)
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

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_rewriting_model(self):
        pipeline_ins = pipeline(
            task=Tasks.text2text_generation,
            model=self.model_id_rewriting,
            model_revision='v1.0.1')
        print(pipeline_ins(self.input_rewriting))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_pipeline_with_model_id_batch(self):
        pipeline_ins = pipeline(
            task=Tasks.text2text_generation, model=self.model_id_translate)
        inputs = [
            self.input_translate, self.input_translate[:8],
            self.input_translate[8:]
        ]
        print(pipeline_ins(inputs, batch_size=2))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_pipeline_with_model_id_batch_iter(self):
        pipeline_ins = pipeline(
            task=Tasks.text2text_generation,
            model=self.model_id_translate,
            padding=False)
        print(
            pipeline_ins([
                self.input_translate, self.input_translate[:8],
                self.input_translate[8:]
            ]))

    @unittest.skip(
        'only for test cases, there is no default official model yet')
    def test_run_pipeline_without_model_id(self):
        pipeline_ins = pipeline(task=Tasks.text2text_generation)
        print(pipeline_ins(self.input_generate))


if __name__ == '__main__':
    unittest.main()
