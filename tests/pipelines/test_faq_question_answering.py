# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import numpy as np

from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import SbertForFaqQuestionAnswering
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import FaqQuestionAnsweringPipeline
from modelscope.preprocessors import FaqQuestionAnsweringPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class FaqQuestionAnsweringTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.faq_question_answering
        self.model_id = 'damo/nlp_structbert_faq-question-answering_chinese-base'

    param = {
        'query_set': ['如何使用优惠券', '在哪里领券', '在哪里领券'],
        'support_set': [{
            'text': '卖品代金券怎么用',
            'label': '6527856'
        }, {
            'text': '怎么使用优惠券',
            'label': '6527856'
        }, {
            'text': '这个可以一起领吗',
            'label': '1000012000'
        }, {
            'text': '付款时送的优惠券哪里领',
            'label': '1000012000'
        }, {
            'text': '购物等级怎么长',
            'label': '13421097'
        }, {
            'text': '购物等级二心',
            'label': '13421097'
        }]
    }

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_direct_file_download(self):
        cache_path = snapshot_download(self.model_id)
        preprocessor = FaqQuestionAnsweringPreprocessor.from_pretrained(
            cache_path)
        model = SbertForFaqQuestionAnswering.from_pretrained(cache_path)
        pipeline_ins = FaqQuestionAnsweringPipeline(
            model, preprocessor=preprocessor)
        result = pipeline_ins(self.param)
        print(result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        preprocessor = FaqQuestionAnsweringPreprocessor(model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.faq_question_answering,
            model=model,
            preprocessor=preprocessor)
        result = pipeline_ins(self.param)
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.faq_question_answering, model=self.model_id)
        result = pipeline_ins(self.param)
        print(result)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.faq_question_answering)
        print(pipeline_ins(self.param, max_seq_length=20))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_sentence_embedding(self):
        pipeline_ins = pipeline(task=Tasks.faq_question_answering)
        sentence_vec = pipeline_ins.get_sentence_embedding(
            ['今天星期六', '明天星期几明天星期几'])
        print(np.shape(sentence_vec))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
