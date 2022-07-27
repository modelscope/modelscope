# Copyright (c) Alibaba, Inc. and its affiliates.
import shutil
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import SbertForTokenClassification
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import WordSegmentationPipeline
from modelscope.preprocessors import TokenClassificationPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class WordSegmentationTest(unittest.TestCase):
    model_id = 'damo/nlp_structbert_word-segmentation_chinese-base'
    sentence = '今天天气不错，适合出去游玩'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        cache_path = snapshot_download(self.model_id)
        tokenizer = TokenClassificationPreprocessor(cache_path)
        model = SbertForTokenClassification(cache_path, tokenizer=tokenizer)
        pipeline1 = WordSegmentationPipeline(model, preprocessor=tokenizer)
        pipeline2 = pipeline(
            Tasks.word_segmentation, model=model, preprocessor=tokenizer)
        print(f'sentence: {self.sentence}\n'
              f'pipeline1:{pipeline1(input=self.sentence)}')
        print()
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
        print(pipeline_ins(input=self.sentence))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.word_segmentation)
        print(pipeline_ins(input=self.sentence))


if __name__ == '__main__':
    unittest.main()
