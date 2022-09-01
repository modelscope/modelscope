# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import SbertForSequenceClassification
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import PairSentenceClassificationPipeline
from modelscope.preprocessors import PairSentenceClassificationPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.regress_test_utils import MsRegressTool
from modelscope.utils.test_utils import test_level


class NLITest(unittest.TestCase):
    model_id = 'damo/nlp_structbert_nli_chinese-base'
    sentence1 = '四川商务职业学院和四川财经职业学院哪个好？'
    sentence2 = '四川商务职业学院商务管理在哪个校区？'
    regress_tool = MsRegressTool(baseline=False)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_direct_file_download(self):
        cache_path = snapshot_download(self.model_id)
        tokenizer = PairSentenceClassificationPreprocessor(cache_path)
        model = SbertForSequenceClassification.from_pretrained(cache_path)
        pipeline1 = PairSentenceClassificationPipeline(
            model, preprocessor=tokenizer)
        pipeline2 = pipeline(Tasks.nli, model=model, preprocessor=tokenizer)
        print(f'sentence1: {self.sentence1}\nsentence2: {self.sentence2}\n'
              f'pipeline1:{pipeline1(input=(self.sentence1, self.sentence2))}')
        print(
            f'sentence1: {self.sentence1}\nsentence2: {self.sentence2}\n'
            f'pipeline1: {pipeline2(input=(self.sentence1, self.sentence2))}')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        tokenizer = PairSentenceClassificationPreprocessor(model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.nli, model=model, preprocessor=tokenizer)
        print(pipeline_ins(input=(self.sentence1, self.sentence2)))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(task=Tasks.nli, model=self.model_id)
        with self.regress_tool.monitor_module_single_forward(
                pipeline_ins.model, 'sbert_nli'):
            print(pipeline_ins(input=(self.sentence1, self.sentence2)))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.nli)
        print(pipeline_ins(input=(self.sentence1, self.sentence2)))


if __name__ == '__main__':
    unittest.main()
