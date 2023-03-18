# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch
from packaging import version

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model, TorchModel
from modelscope.models.nlp import SbertForSequenceClassification
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import TextClassificationPipeline
from modelscope.preprocessors import TextClassificationTransformersPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.regress_test_utils import IgnoreKeyFn, MsRegressTool
from modelscope.utils.test_utils import test_level


class SentenceSimilarityTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.sentence_similarity
        self.model_id = 'damo/nlp_structbert_sentence-similarity_chinese-base'
        self.model_id_retail = 'damo/nlp_structbert_sentence-similarity_chinese-retail-base'

    sentence1 = '今天气温比昨天高么？'
    sentence2 = '今天湿度比昨天高么？'
    regress_tool = MsRegressTool(baseline=False)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run(self):
        cache_path = snapshot_download(self.model_id)
        tokenizer = TextClassificationTransformersPreprocessor(cache_path)
        model = SbertForSequenceClassification.from_pretrained(cache_path)
        pipeline1 = TextClassificationPipeline(model, preprocessor=tokenizer)
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
        tokenizer = TextClassificationTransformersPreprocessor(model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.sentence_similarity,
            model=model,
            preprocessor=tokenizer)
        print(pipeline_ins(input=(self.sentence1, self.sentence2)))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_batch(self):
        pipeline_ins = pipeline(
            task=Tasks.sentence_similarity, model=self.model_id)
        print(
            pipeline_ins(
                input=[(self.sentence1, self.sentence2),
                       (self.sentence1[:4], self.sentence2[5:]),
                       (self.sentence1[2:], self.sentence2[:8])],
                batch_size=2))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_batch_iter(self):
        pipeline_ins = pipeline(
            task=Tasks.sentence_similarity, model=self.model_id, padding=False)
        print(
            pipeline_ins(input=[(
                self.sentence1,
                self.sentence2), (self.sentence1[:4], self.sentence2[5:]
                                  ), (self.sentence1[2:],
                                      self.sentence2[:8])]))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.sentence_similarity, model=self.model_id)
        with self.regress_tool.monitor_module_single_forward(
                pipeline_ins.model,
                'sbert_sen_sim',
                compare_fn=IgnoreKeyFn('.*intermediate_act_fn')):
            print(pipeline_ins(input=(self.sentence1, self.sentence2)))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_retail_similarity_model(self):
        pipeline_ins = pipeline(
            task=Tasks.sentence_similarity,
            model=self.model_id_retail,
            model_revision='v1.0.0')
        print(pipeline_ins(input=(self.sentence1, self.sentence2)))

    @unittest.skipIf(
        version.parse(torch.__version__) < version.parse('2.0.0.dev'),
        'skip when torch version < 2.0')
    def test_compile(self):
        pipeline_ins = pipeline(
            task=Tasks.sentence_similarity,
            model=self.model_id_retail,
            model_revision='v1.0.0',
            compile=True)
        print(pipeline_ins(input=(self.sentence1, self.sentence2)))
        self.assertTrue(isinstance(pipeline_ins.model._orig_mod, TorchModel))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.sentence_similarity)
        print(pipeline_ins(input=(self.sentence1, self.sentence2)))

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
