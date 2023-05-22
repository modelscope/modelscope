# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import ModelForTokenClassificationWithCRF
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import WordSegmentationThaiPipeline
from modelscope.preprocessors import WordSegmentationPreprocessorThai
from modelscope.utils.constant import Tasks
from modelscope.utils.regress_test_utils import MsRegressTool
from modelscope.utils.test_utils import test_level


class WordSegmentationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.word_segmentation
        self.model_id = 'damo/nlp_xlmr_word-segmentation_thai'

    sentence = 'รถคันเก่าก็ยังเก็บเอาไว้ยังไม่ได้ขาย'
    regress_tool = MsRegressTool(baseline=False)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        tokenizer = WordSegmentationPreprocessorThai(cache_path)
        model = ModelForTokenClassificationWithCRF.from_pretrained(cache_path)
        pipeline1 = WordSegmentationThaiPipeline(model, preprocessor=tokenizer)
        pipeline2 = pipeline(
            Tasks.word_segmentation, model=model, preprocessor=tokenizer)
        print(f'sentence: {self.sentence}\n'
              f'pipeline1:{pipeline1(input=self.sentence)}')
        print(f'pipeline2: {pipeline2(input=self.sentence)}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        tokenizer = WordSegmentationPreprocessorThai(model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.word_segmentation, model=model, preprocessor=tokenizer)
        print(pipeline_ins(input=self.sentence))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.word_segmentation, model=self.model_id)
        print(pipeline_ins(input=self.sentence))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_batch(self):
        pipeline_ins = pipeline(
            task=Tasks.word_segmentation, model=self.model_id)
        print(
            pipeline_ins(
                input=[self.sentence, self.sentence[:10], self.sentence[6:]],
                batch_size=2))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name_batch_iter(self):
        pipeline_ins = pipeline(
            task=Tasks.word_segmentation, model=self.model_id, padding=False)
        print(
            pipeline_ins(
                input=[self.sentence, self.sentence[:10], self.sentence[6:]]))


if __name__ == '__main__':
    unittest.main()
