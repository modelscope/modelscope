# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import TextClassificationPipeline
from modelscope.preprocessors import TextClassificationTransformersPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.regress_test_utils import IgnoreKeyFn, MsRegressTool
from modelscope.utils.test_utils import test_level


class NLITest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.nli
        self.model_id = 'damo/nlp_structbert_nli_chinese-base'
        self.model_id_fact_checking = 'damo/nlp_structbert_fact-checking_chinese-base'
        self.model_id_peer = 'damo/nlp_peer_mnli_english-base'

    sentence1 = '四川商务职业学院和四川财经职业学院哪个好？'
    sentence2 = '四川商务职业学院商务管理在哪个校区？'
    en_sentence1 = 'Conceptually cream skimming has two basic dimensions - product and geography.'
    en_sentence2 = 'Product and geography are what make cream skimming work.'
    regress_tool = MsRegressTool(baseline=False)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_direct_file_download(self):
        cache_path = snapshot_download(self.model_id)
        tokenizer = TextClassificationTransformersPreprocessor(cache_path)
        model = Model.from_pretrained(cache_path)
        pipeline1 = TextClassificationPipeline(model, preprocessor=tokenizer)
        pipeline2 = pipeline(Tasks.nli, model=model, preprocessor=tokenizer)
        print(f'sentence1: {self.sentence1}\nsentence2: {self.sentence2}\n'
              f'pipeline1:{pipeline1(input=(self.sentence1, self.sentence2))}')
        print(
            f'sentence1: {self.sentence1}\nsentence2: {self.sentence2}\n'
            f'pipeline1: {pipeline2(input=(self.sentence1, self.sentence2))}')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        tokenizer = TextClassificationTransformersPreprocessor(model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.nli, model=model, preprocessor=tokenizer)
        print(pipeline_ins(input=(self.sentence1, self.sentence2)))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(task=Tasks.nli, model=self.model_id)
        with self.regress_tool.monitor_module_single_forward(
                pipeline_ins.model,
                'sbert_nli',
                compare_fn=IgnoreKeyFn('.*intermediate_act_fn')):
            print(pipeline_ins(input=(self.sentence1, self.sentence2)))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_fact_checking_model(self):
        pipeline_ins = pipeline(
            task=Tasks.nli,
            model=self.model_id_fact_checking,
            model_revision='v1.0.1')
        print(pipeline_ins(input=(self.sentence1, self.sentence2)))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_peer_model(self):
        pipeline_ins = pipeline(
            task=Tasks.nli,
            model=self.model_id_peer,
            model_revision='v1.0.0',
        )
        print(pipeline_ins(input=(self.en_sentence1, self.en_sentence2)))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.nli)
        print(pipeline_ins(input=(self.sentence1, self.sentence2)))


if __name__ == '__main__':
    unittest.main()
