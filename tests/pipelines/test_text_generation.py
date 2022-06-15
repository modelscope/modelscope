# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from maas_hub.snapshot_download import snapshot_download

from modelscope.models import Model
from modelscope.models.nlp import PalmForTextGenerationModel
from modelscope.pipelines import TextGenerationPipeline, pipeline
from modelscope.preprocessors import TextGenerationPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class TextGenerationTest(unittest.TestCase):
    model_id = 'damo/nlp_palm_text-generation_chinese'
    input1 = "今日天气类型='晴'&温度变化趋势='大幅上升'&最低气温='28℃'&最高气温='31℃'&体感='湿热'"
    input2 = "今日天气类型='多云'&体感='舒适'&最低气温='26℃'&最高气温='30℃'"

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run(self):
        cache_path = snapshot_download(self.model_id)
        preprocessor = TextGenerationPreprocessor(
            cache_path, first_sequence='sentence', second_sequence=None)
        model = PalmForTextGenerationModel(
            cache_path, tokenizer=preprocessor.tokenizer)
        pipeline1 = TextGenerationPipeline(model, preprocessor)
        pipeline2 = pipeline(
            Tasks.text_generation, model=model, preprocessor=preprocessor)
        print(f'input: {self.input1}\npipeline1: {pipeline1(self.input1)}')
        print()
        print(f'input: {self.input2}\npipeline2: {pipeline2(self.input2)}')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        preprocessor = TextGenerationPreprocessor(
            model.model_dir, first_sequence='sentence', second_sequence=None)
        pipeline_ins = pipeline(
            task=Tasks.text_generation, model=model, preprocessor=preprocessor)
        print(pipeline_ins(self.input1))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.text_generation, model=self.model_id)
        print(pipeline_ins(self.input2))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.text_generation)
        print(pipeline_ins(self.input2))


if __name__ == '__main__':
    unittest.main()
