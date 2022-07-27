# Copyright (c) Alibaba, Inc. and its affiliates.
import shutil
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import SbertForSentenceSimilarity
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import SentenceSimilarityPipeline
from modelscope.preprocessors import SentenceSimilarityPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class SentenceSimilarityTest(unittest.TestCase):
    model_id = 'damo/nlp_structbert_sentence-similarity_chinese-base'
    sentence1 = '今天气温比昨天高么？'
    sentence2 = '今天湿度比昨天高么？'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run(self):
        cache_path = snapshot_download(self.model_id)
        tokenizer = SentenceSimilarityPreprocessor(cache_path)
        model = SbertForSentenceSimilarity(cache_path, tokenizer=tokenizer)
        pipeline1 = SentenceSimilarityPipeline(model, preprocessor=tokenizer)
        pipeline2 = pipeline(
            Tasks.sentence_similarity, model=model, preprocessor=tokenizer)
        print('test1')
        print(f'sentence1: {self.sentence1}\nsentence2: {self.sentence2}\n'
              f'pipeline1:{pipeline1(input=(self.sentence1, self.sentence2))}')
        print()
        print(
            f'sentence1: {self.sentence1}\nsentence2: {self.sentence2}\n'
            f'pipeline1: {pipeline2(input=(self.sentence1, self.sentence2))}')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        tokenizer = SentenceSimilarityPreprocessor(model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.sentence_similarity,
            model=model,
            preprocessor=tokenizer)
        print(pipeline_ins(input=(self.sentence1, self.sentence2)))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.sentence_similarity, model=self.model_id)
        print(pipeline_ins(input=(self.sentence1, self.sentence2)))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.sentence_similarity)
        print(pipeline_ins(input=(self.sentence1, self.sentence2)))


if __name__ == '__main__':
    unittest.main()
