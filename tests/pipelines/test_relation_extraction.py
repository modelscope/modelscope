# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import ModelForInformationExtraction
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import InformationExtractionPipeline
from modelscope.preprocessors import RelationExtractionTransformersPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class RelationExtractionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.relation_extraction
        self.model_id = 'damo/nlp_bert_relation-extraction_chinese-base'

    sentence = '高捷，祖籍江苏，本科毕业于东南大学'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        tokenizer = RelationExtractionTransformersPreprocessor(cache_path)
        model = ModelForInformationExtraction.from_pretrained(cache_path)
        pipeline1 = InformationExtractionPipeline(
            model, preprocessor=tokenizer)
        pipeline2 = pipeline(
            Tasks.relation_extraction, model=model, preprocessor=tokenizer)
        print(f'sentence: {self.sentence}\n'
              f'pipeline1:{pipeline1(input=self.sentence)}')
        print()
        print(f'pipeline2: {pipeline2(input=self.sentence)}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        tokenizer = RelationExtractionTransformersPreprocessor(model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.relation_extraction,
            model=model,
            preprocessor=tokenizer)
        print(pipeline_ins(input=self.sentence))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.relation_extraction, model=self.model_id)
        print(pipeline_ins(input=self.sentence))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.relation_extraction)
        print(pipeline_ins(input=self.sentence))


if __name__ == '__main__':
    unittest.main()
