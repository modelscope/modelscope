# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import SbertForSequenceClassification
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import SingleSentenceClassificationPipeline
from modelscope.preprocessors import SingleSentenceClassificationPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class SentimentClassificationTest(unittest.TestCase):
    model_id = 'damo/nlp_structbert_sentiment-classification_chinese-tiny'
    sentence1 = '启动的时候很大声音，然后就会听到1.2秒的卡察的声音，类似齿轮摩擦的声音'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_direct_file_download(self):
        cache_path = snapshot_download(self.model_id)
        tokenizer = SingleSentenceClassificationPreprocessor(cache_path)
        model = SbertForSequenceClassification.from_pretrained(
            self.model_id, num_labels=2)
        pipeline1 = SingleSentenceClassificationPipeline(
            model, preprocessor=tokenizer)
        pipeline2 = pipeline(
            Tasks.sentiment_classification,
            model=model,
            preprocessor=tokenizer)
        print(f'sentence1: {self.sentence1}\n'
              f'pipeline1:{pipeline1(input=self.sentence1)}')
        print()
        print(f'sentence1: {self.sentence1}\n'
              f'pipeline1: {pipeline2(input=self.sentence1)}')
        self.assertTrue(
            isinstance(pipeline1.model, SbertForSequenceClassification))
        self.assertTrue(
            isinstance(pipeline2.model, SbertForSequenceClassification))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        tokenizer = SingleSentenceClassificationPreprocessor(model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.sentiment_classification,
            model=model,
            preprocessor=tokenizer)
        print(pipeline_ins(input=self.sentence1))
        self.assertTrue(
            isinstance(pipeline_ins.model, SbertForSequenceClassification))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.sentiment_classification, model=self.model_id)
        print(pipeline_ins(input=self.sentence1))
        print(pipeline_ins.model.__class__)
        self.assertTrue(
            isinstance(pipeline_ins.model, SbertForSequenceClassification))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.sentiment_classification)
        print(pipeline_ins(input=self.sentence1))
        self.assertTrue(
            isinstance(pipeline_ins.model, SbertForSequenceClassification))


if __name__ == '__main__':
    unittest.main()
