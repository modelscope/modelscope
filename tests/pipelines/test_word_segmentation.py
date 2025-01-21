# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import (LSTMForTokenClassificationWithCRF,
                                   SbertForTokenClassification)
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import WordSegmentationPipeline
from modelscope.preprocessors import \
    TokenClassificationTransformersPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.regress_test_utils import IgnoreKeyFn, MsRegressTool
from modelscope.utils.test_utils import test_level


class WordSegmentationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.word_segmentation
        self.model_id = 'damo/nlp_structbert_word-segmentation_chinese-base'
        self.ecom_model_id = 'damo/nlp_structbert_word-segmentation_chinese-base-ecommerce'
        self.lstmcrf_news_model_id = 'damo/nlp_lstmcrf_word-segmentation_chinese-news'
        self.lstmcrf_ecom_model_id = 'damo/nlp_lstmcrf_word-segmentation_chinese-ecommerce'

    sentence = '今天天气不错，适合出去游玩'
    sentence_ecom = '东阳草肌醇复合物'
    sentence_eng = 'I am a program.'
    regress_tool = MsRegressTool(baseline=False)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        tokenizer = TokenClassificationTransformersPreprocessor(cache_path)
        model = SbertForTokenClassification.from_pretrained(cache_path)
        pipeline1 = WordSegmentationPipeline(model, preprocessor=tokenizer)
        pipeline2 = pipeline(
            Tasks.word_segmentation, model=model, preprocessor=tokenizer)
        print(f'sentence: {self.sentence}\n'
              f'pipeline1:{pipeline1(input=self.sentence)}')
        print(f'pipeline2: {pipeline2(input=self.sentence)}')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_ecom_by_direct_model_download(self):
        cache_path = snapshot_download(self.ecom_model_id)
        tokenizer = TokenClassificationTransformersPreprocessor(cache_path)
        model = SbertForTokenClassification.from_pretrained(cache_path)
        pipeline1 = WordSegmentationPipeline(model, preprocessor=tokenizer)
        pipeline2 = pipeline(
            Tasks.word_segmentation, model=model, preprocessor=tokenizer)
        print(f'sentence: {self.sentence_ecom}\n'
              f'pipeline1:{pipeline1(input=self.sentence_ecom)}')
        print(f'pipeline2: {pipeline2(input=self.sentence_ecom)}')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_lstmcrf_news_by_direct_model_download(self):
        cache_path = snapshot_download(self.lstmcrf_news_model_id)
        tokenizer = TokenClassificationTransformersPreprocessor(cache_path)
        model = LSTMForTokenClassificationWithCRF.from_pretrained(cache_path)
        pipeline1 = WordSegmentationPipeline(model, preprocessor=tokenizer)
        pipeline2 = pipeline(
            Tasks.word_segmentation, model=model, preprocessor=tokenizer)
        print(f'sentence: {self.sentence}\n'
              f'pipeline1:{pipeline1(input=self.sentence)}')
        print(f'pipeline2: {pipeline2(input=self.sentence)}')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_lstmcrf_ecom_by_direct_model_download(self):
        cache_path = snapshot_download(self.lstmcrf_ecom_model_id)
        tokenizer = TokenClassificationTransformersPreprocessor(cache_path)
        model = LSTMForTokenClassificationWithCRF.from_pretrained(cache_path)
        pipeline1 = WordSegmentationPipeline(model, preprocessor=tokenizer)
        pipeline2 = pipeline(
            Tasks.word_segmentation, model=model, preprocessor=tokenizer)
        print(f'sentence: {self.sentence_ecom}\n'
              f'pipeline1:{pipeline1(input=self.sentence_ecom)}')
        print(f'pipeline2: {pipeline2(input=self.sentence_ecom)}')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        tokenizer = TokenClassificationTransformersPreprocessor(
            model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.word_segmentation, model=model, preprocessor=tokenizer)
        print(pipeline_ins(input=self.sentence))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_ecom_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.ecom_model_id)
        tokenizer = TokenClassificationTransformersPreprocessor(
            model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.word_segmentation, model=model, preprocessor=tokenizer)
        print(pipeline_ins(input=self.sentence_ecom))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_lstmcrf_news_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.lstmcrf_news_model_id)
        tokenizer = TokenClassificationTransformersPreprocessor(
            model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.word_segmentation, model=model, preprocessor=tokenizer)
        print(pipeline_ins(input=self.sentence))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_lstmcrf_ecom_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.lstmcrf_ecom_model_id)
        tokenizer = TokenClassificationTransformersPreprocessor(
            model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.word_segmentation, model=model, preprocessor=tokenizer)
        print(pipeline_ins(input=self.sentence_ecom))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.word_segmentation, model=self.model_id)
        with self.regress_tool.monitor_module_single_forward(
                pipeline_ins.model,
                'sbert_ws_zh',
                compare_fn=IgnoreKeyFn('.*intermediate_act_fn')):
            print(pipeline_ins(input=self.sentence))
        print(pipeline_ins(input=self.sentence_eng))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_ecom_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.word_segmentation, model=self.ecom_model_id)
        print(pipeline_ins(input=self.sentence_ecom))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_lstmcrf_news_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.word_segmentation, model=self.lstmcrf_news_model_id)
        print(pipeline_ins(input=self.sentence))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_lstmcrf_ecom_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.word_segmentation, model=self.lstmcrf_ecom_model_id)
        print(pipeline_ins(input=self.sentence_ecom))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_batch(self):
        pipeline_ins = pipeline(
            task=Tasks.word_segmentation, model=self.model_id)
        print(
            pipeline_ins(
                input=[self.sentence, self.sentence[:5], self.sentence[5:]],
                batch_size=2))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_batch_iter(self):
        pipeline_ins = pipeline(
            task=Tasks.word_segmentation, model=self.model_id, padding=False)
        print(
            pipeline_ins(
                input=[self.sentence, self.sentence[:5], self.sentence[5:]]))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.word_segmentation)
        print(pipeline_ins(input=self.sentence))


if __name__ == '__main__':
    unittest.main()
