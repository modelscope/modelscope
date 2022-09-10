# Copyright (c) Alibaba, Inc. and its affiliates.
import shutil
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import PassageRanking
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import PassageRankingPipeline
from modelscope.preprocessors import PassageRankingPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class PassageRankingTest(unittest.TestCase):
    model_id = 'damo/nlp_corom_passage-ranking_english-base'
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
        cache_path = snapshot_download(self.model_id)
        tokenizer = PassageRankingPreprocessor(cache_path)
        model = PassageRanking.from_pretrained(cache_path)
        pipeline1 = PassageRankingPipeline(model, preprocessor=tokenizer)
        pipeline2 = pipeline(
            Tasks.passage_ranking, model=model, preprocessor=tokenizer)
        print(f'sentence: {self.inputs}\n'
              f'pipeline1:{pipeline1(input=self.inputs)}')
        print()
        print(f'pipeline2: {pipeline2(input=self.inputs)}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        tokenizer = PassageRankingPreprocessor(model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.passage_ranking, model=model, preprocessor=tokenizer)
        print(pipeline_ins(input=self.inputs))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.passage_ranking, model=self.model_id)
        print(pipeline_ins(input=self.inputs))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.passage_ranking)
        print(pipeline_ins(input=self.inputs))


if __name__ == '__main__':
    unittest.main()
