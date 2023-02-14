# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import SiameseUieModel
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import SiameseUiePipeline
from modelscope.preprocessors import SiameseUiePreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.regress_test_utils import IgnoreKeyFn, MsRegressTool
from modelscope.utils.test_utils import test_level


class ZeroShotClassificationTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.siamese_uie
        self.model_id = 'damo/nlp_structbert_siamese-uie_chinese-base'

    sentence = '1944年毕业于北大的名古屋铁道会长谷口清太郎等人在日本积极筹资，共筹款2.7亿日元，参加捐款的日本企业有69家。'
    schema = {'人物': None, '地理位置': None, '组织机构': None}
    regress_tool = MsRegressTool(baseline=False)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_direct_file_download(self):
        cache_path = snapshot_download(self.model_id)
        tokenizer = SiameseUiePreprocessor(cache_path)
        model = SiameseUieModel.from_pretrained(cache_path)
        pipeline1 = SiameseUiePipeline(
            model, preprocessor=tokenizer, model_revision='v1.0')
        pipeline2 = pipeline(
            Tasks.siamese_uie,
            model=model,
            preprocessor=tokenizer,
            model_revision='v1.0')

        print(
            f'sentence: {self.sentence}\n'
            f'pipeline1:{pipeline1(input=self.sentence, schema=self.schema)}')
        print(f'sentence: {self.sentence}\n'
              f'pipeline2: {pipeline2(self.sentence, schema=self.schema)}')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        tokenizer = SiameseUiePreprocessor(model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.siamese_uie,
            model=model,
            preprocessor=tokenizer,
            model_revision='v1.0')
        print(pipeline_ins(input=self.sentence, schema=self.schema))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.siamese_uie, model=self.model_id, model_revision='v1.0')
        print(pipeline_ins(input=self.sentence, schema=self.schema))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.siamese_uie, model_revision='v1.0')
        print(pipeline_ins(input=self.sentence, schema=self.schema))

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
