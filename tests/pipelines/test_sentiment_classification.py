# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp.task_models.sequence_classification import \
    SequenceClassificationModel
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import TextClassificationPipeline
from modelscope.preprocessors import TextClassificationTransformersPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class SentimentClassificationTaskModelTest(unittest.TestCase,
                                           DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.text_classification
        self.model_id = 'damo/nlp_structbert_sentiment-classification_chinese-base'

    sentence1 = '启动的时候很大声音，然后就会听到1.2秒的卡察的声音，类似齿轮摩擦的声音'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_direct_file_download(self):
        cache_path = snapshot_download(self.model_id)
        tokenizer = TextClassificationTransformersPreprocessor(cache_path)
        model = SequenceClassificationModel.from_pretrained(
            self.model_id, num_labels=2)
        pipeline1 = TextClassificationPipeline(model, preprocessor=tokenizer)
        pipeline2 = pipeline(
            Tasks.text_classification, model=model, preprocessor=tokenizer)
        print(f'sentence1: {self.sentence1}\n'
              f'pipeline1:{pipeline1(input=self.sentence1)}')
        print(f'sentence1: {self.sentence1}\n'
              f'pipeline1: {pipeline2(input=self.sentence1)}')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        tokenizer = TextClassificationTransformersPreprocessor(model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.text_classification,
            model=model,
            preprocessor=tokenizer)
        print(pipeline_ins(input=self.sentence1))
        self.assertTrue(
            isinstance(pipeline_ins.model, SequenceClassificationModel))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.text_classification, model=self.model_id)
        print(pipeline_ins(input=self.sentence1))
        self.assertTrue(
            isinstance(pipeline_ins.model, SequenceClassificationModel))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.text_classification)
        print(pipeline_ins(input=self.sentence1))
        self.assertTrue(
            isinstance(pipeline_ins.model, SequenceClassificationModel))

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
