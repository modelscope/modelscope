# Copyright (c) Alibaba, Inc. and its affiliates.
import shutil
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import BertForTextRanking
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import TextRankingPipeline
from modelscope.preprocessors import TextRankingPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class TextRankingTest(unittest.TestCase):
    models = [
        'damo/nlp_corom_passage-ranking_english-base',
        'damo/nlp_rom_passage-ranking_chinese-base'
    ]

    inputs = {
        'source_sentence': ["how long it take to get a master's degree"],
        'sentences_to_compare': [
            "On average, students take about 18 to 24 months to complete a master's degree.",
            'On the other hand, some students prefer to go at a slower pace and choose to take '
            'several years to complete their studies.',
            'It can take anywhere from two semesters'
        ]
    }

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        for model_id in self.models:
            cache_path = snapshot_download(model_id)
            tokenizer = TextRankingPreprocessor(cache_path)
            model = BertForTextRanking.from_pretrained(cache_path)
            pipeline1 = TextRankingPipeline(model, preprocessor=tokenizer)
            pipeline2 = pipeline(
                Tasks.text_ranking, model=model, preprocessor=tokenizer)
            print(f'sentence: {self.inputs}\n'
                  f'pipeline1:{pipeline1(input=self.inputs)}')
            print()
            print(f'pipeline2: {pipeline2(input=self.inputs)}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        for model_id in self.models:
            model = Model.from_pretrained(model_id)
            tokenizer = TextRankingPreprocessor(model.model_dir)
            pipeline_ins = pipeline(
                task=Tasks.text_ranking, model=model, preprocessor=tokenizer)
            print(pipeline_ins(input=self.inputs))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_name(self):
        for model_id in self.models:
            pipeline_ins = pipeline(task=Tasks.text_ranking, model=model_id)
            print(pipeline_ins(input=self.inputs))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.text_ranking)
        print(pipeline_ins(input=self.inputs))


if __name__ == '__main__':
    unittest.main()
