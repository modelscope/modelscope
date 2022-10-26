# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import SbertForFaqRanking, SbertForFaqRetrieval
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import FaqPipeline
from modelscope.preprocessors import FaqPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class FaqTest(unittest.TestCase):
    model_id = '/Users/tanfan/Desktop/Workdir/Gitlab/maas/MaaS-lib/.faq_test_model'
    param = {
        'query_set': ['明天星期几', '今天星期六', '今天星期六'],
        'support_set': [{
            'text': '今天星期六',
            'label': 'label0'
        }, {
            'text': '明天星期几',
            'label': 'label1'
        }]
    }

    # @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    # def test_run_with_direct_file_download(self):
    #     cache_path = self.model_id  # snapshot_download(self.model_id)
    #     preprocessor = FaqPreprocessor(cache_path)
    #     model = SbertForFaq(cache_path)
    #     pipeline_ins = FaqPipeline(model, preprocessor=preprocessor)
    #
    #     result = pipeline_ins(self.param)
    #     print(result)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        preprocessor = FaqPreprocessor(model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.faq, model=model, preprocessor=preprocessor)
        result = pipeline_ins(self.param)
        print(result)

    # @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    # def test_run_with_model_name(self):
    #     pipeline_ins = pipeline(task=Tasks.faq, model=self.model_id)
    #     result = pipeline_ins(self.param)
    #     print(result)

    # @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    # def test_run_with_default_model(self):
    #     pipeline_ins = pipeline(task=Tasks.faq)
    #     print(pipeline_ins(self.param))


if __name__ == '__main__':
    unittest.main()
