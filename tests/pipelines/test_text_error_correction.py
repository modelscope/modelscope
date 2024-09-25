# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import BartForTextErrorCorrection
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import TextErrorCorrectionPipeline
from modelscope.preprocessors import (Preprocessor,
                                      TextErrorCorrectionPreprocessor)
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class TextErrorCorrectionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.text_error_correction
        self.model_id = 'damo/nlp_bart_text-error-correction_chinese'
        self.law_model_id = 'damo/nlp_bart_text-error-correction_chinese-law'

    input = '随着中国经济突飞猛近，建造工业与日俱增'
    input_2 = '这洋的话，下一年的福气来到自己身上。'
    input_3 = '在拥挤时间，为了让人们尊守交通规律，派至少两个警察或者交通管理者。'
    input_4 = 'LSTM在拥挤时间，为了让人们尊守交通规律，派至少两个警察或者交通管理者。'
    input_law = '2012年、2013年收入统计表复印件各一份，欲证明被告未足额知府社保费用。'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_direct_download(self):
        cache_path = snapshot_download(self.model_id)
        model = BartForTextErrorCorrection(cache_path)
        preprocessor = Preprocessor.from_pretrained(cache_path)
        pipeline1 = TextErrorCorrectionPipeline(model, preprocessor)
        pipeline2 = pipeline(
            Tasks.text_error_correction,
            model=model,
            preprocessor=preprocessor)
        print(
            f'pipeline1: {pipeline1(self.input)}\npipeline2: {pipeline2(self.input)}'
        )

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_batch(self):
        pipeline_ins = pipeline(
            task=Tasks.text_error_correction, model=self.model_id)
        sents = [self.input, self.input_2, self.input_3, self.input_4, self.input_law]
        rs1 = pipeline_ins(sents, batch_size=2)
        rs2 = pipeline_ins(sents)
        print('batch: ', rs1, rs2)
        self.assertEqual(rs1, rs2)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        preprocessor = Preprocessor.from_pretrained(model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.text_error_correction,
            model=model,
            preprocessor=preprocessor)
        print(pipeline_ins(self.input))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.text_error_correction, model=self.model_id)
        print(pipeline_ins(self.input))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.text_error_correction)
        print(pipeline_ins(self.input))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_english_input(self):
        pipeline_ins = pipeline(task=Tasks.text_error_correction)
        print(pipeline_ins(self.input_4))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_law_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.text_error_correction, model=self.law_model_id)
        print(pipeline_ins(self.input_law))


if __name__ == '__main__':
    unittest.main()
