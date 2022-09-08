# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import SbertForTokenClassification
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import WordSegmentationPipeline
from modelscope.preprocessors import TokenClassificationPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.regress_test_utils import MsRegressTool
from modelscope.utils.test_utils import test_level


class WordSegmentationTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.word_segmentation
        self.model_id = 'damo/nlp_structbert_word-segmentation_chinese-base'

    sentence = '今天天气不错，适合出去游玩'
    sentence_eng = 'I am a program.'
    regress_tool = MsRegressTool(baseline=False)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        tokenizer = TokenClassificationPreprocessor(cache_path)
        model = SbertForTokenClassification.from_pretrained(cache_path)
        pipeline1 = WordSegmentationPipeline(model, preprocessor=tokenizer)
        pipeline2 = pipeline(
            Tasks.word_segmentation, model=model, preprocessor=tokenizer)
        print(f'sentence: {self.sentence}\n'
              f'pipeline1:{pipeline1(input=self.sentence)}')
        print(f'pipeline2: {pipeline2(input=self.sentence)}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        tokenizer = TokenClassificationPreprocessor(model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.word_segmentation, model=model, preprocessor=tokenizer)
        print(pipeline_ins(input=self.sentence))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.word_segmentation, model=self.model_id)
        with self.regress_tool.monitor_module_single_forward(
                pipeline_ins.model, 'sbert_ws_zh'):
            print(pipeline_ins(input=self.sentence))
        with self.regress_tool.monitor_module_single_forward(
                pipeline_ins.model, 'sbert_ws_en'):
            print(pipeline_ins(input=self.sentence_eng))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.word_segmentation)
        print(pipeline_ins(input=self.sentence))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
