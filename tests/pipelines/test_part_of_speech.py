# Copyright (c) Alibaba, Inc. and its affiliates.
import shutil
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import (LSTMForTokenClassificationWithCRF,
                                   ModelForTokenClassification)
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import TokenClassificationPipeline
from modelscope.preprocessors import \
    TokenClassificationTransformersPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class PartOfSpeechTest(unittest.TestCase):
    model_id = 'damo/nlp_structbert_part-of-speech_chinese-lite'
    lstmcrf_news_model_id = 'damo/nlp_lstmcrf_part-of-speech_chinese-news'
    sentence = '今天天气不错，适合出去游玩'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        tokenizer = TokenClassificationTransformersPreprocessor(cache_path)
        model = ModelForTokenClassification.from_pretrained(cache_path)
        pipeline1 = TokenClassificationPipeline(model, preprocessor=tokenizer)
        pipeline2 = pipeline(
            Tasks.part_of_speech, model=model, preprocessor=tokenizer)
        print(f'sentence: {self.sentence}\n'
              f'pipeline1:{pipeline1(input=self.sentence)}')
        print()
        print(f'pipeline2: {pipeline2(input=self.sentence)}')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_lstmcrf_news_by_direct_model_download(self):
        cache_path = snapshot_download(self.lstmcrf_news_model_id)
        tokenizer = TokenClassificationTransformersPreprocessor(cache_path)
        model = LSTMForTokenClassificationWithCRF.from_pretrained(cache_path)
        pipeline1 = TokenClassificationPipeline(model, preprocessor=tokenizer)
        pipeline2 = pipeline(
            Tasks.part_of_speech, model=model, preprocessor=tokenizer)
        print(f'sentence: {self.sentence}\n'
              f'pipeline1:{pipeline1(input=self.sentence)}')
        print()
        print(f'pipeline2: {pipeline2(input=self.sentence)}')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        tokenizer = TokenClassificationTransformersPreprocessor(
            model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.part_of_speech, model=model, preprocessor=tokenizer)
        print(pipeline_ins(input=self.sentence))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_lstmcrf_news_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.lstmcrf_news_model_id)
        tokenizer = TokenClassificationTransformersPreprocessor(
            model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.part_of_speech, model=model, preprocessor=tokenizer)
        print(pipeline_ins(input=self.sentence))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(task=Tasks.part_of_speech, model=self.model_id)
        print(pipeline_ins(input=self.sentence))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_lstmcrf_new_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.part_of_speech, model=self.lstmcrf_news_model_id)
        print(pipeline_ins(input=self.sentence))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.part_of_speech)
        print(pipeline_ins(input=self.sentence))


if __name__ == '__main__':
    unittest.main()
