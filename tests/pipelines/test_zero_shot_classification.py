# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from maas_hub.snapshot_download import snapshot_download

from modelscope.models import Model
from modelscope.models.nlp import BertForZeroShotClassification
from modelscope.pipelines import ZeroShotClassificationPipeline, pipeline
from modelscope.preprocessors import ZeroShotClassificationPreprocessor
from modelscope.utils.constant import Tasks


class ZeroShotClassificationTest(unittest.TestCase):
    model_id = 'damo/nlp_structbert_zero-shot-classification_chinese-base'
    sentence = '全新突破 解放军运20版空中加油机曝光'
    labels = ['文化', '体育', '娱乐', '财经', '家居', '汽车', '教育', '科技', '军事']
    template = '这篇文章的标题是{}'

    def test_run_from_local(self):
        cache_path = snapshot_download(self.model_id)
        tokenizer = ZeroShotClassificationPreprocessor(cache_path)
        model = BertForZeroShotClassification(cache_path, tokenizer=tokenizer)
        pipeline1 = ZeroShotClassificationPipeline(
            model, preprocessor=tokenizer)
        pipeline2 = pipeline(
            Tasks.zero_shot_classification,
            model=model,
            preprocessor=tokenizer)

        print(
            f'sentence: {self.sentence}\n'
            f'pipeline1:{pipeline1(input=self.sentence,candidate_labels=self.labels)}'
        )
        print()
        print(
            f'sentence: {self.sentence}\n'
            f'pipeline2: {pipeline2(self.sentence,candidate_labels=self.labels,hypothesis_template=self.template)}'
        )

    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        tokenizer = ZeroShotClassificationPreprocessor(model.model_dir)
        pipeline_ins = pipeline(
            task=Tasks.zero_shot_classification,
            model=model,
            preprocessor=tokenizer)
        print(pipeline_ins(input=self.sentence, candidate_labels=self.labels))

    def test_run_with_model_name(self):
        pipeline_ins = pipeline(
            task=Tasks.zero_shot_classification, model=self.model_id)
        print(pipeline_ins(input=self.sentence, candidate_labels=self.labels))

    def test_run_with_default_model(self):
        pipeline_ins = pipeline(task=Tasks.zero_shot_classification)
        print(pipeline_ins(input=self.sentence, candidate_labels=self.labels))


if __name__ == '__main__':
    unittest.main()
