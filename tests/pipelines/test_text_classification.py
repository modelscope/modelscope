# Copyright (c) Alibaba, Inc. and its affiliates.
import shutil
import unittest
import zipfile
from pathlib import Path

from modelscope.fileio import File
from modelscope.models import Model
from modelscope.models.nlp import BertForSequenceClassification
from modelscope.pipelines import SequenceClassificationPipeline, pipeline
from modelscope.preprocessors import SequenceClassificationPreprocessor
from modelscope.pydatasets import PyDataset
from modelscope.utils.constant import Hubs, Tasks
from modelscope.utils.test_utils import test_level


class SequenceClassificationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/bert-base-sst2'

    def predict(self, pipeline_ins: SequenceClassificationPipeline):
        from easynlp.appzoo import load_dataset

        set = load_dataset('glue', 'sst2')
        data = set['test']['sentence'][:3]

        results = pipeline_ins(data[0])
        print(results)
        results = pipeline_ins(data[1])
        print(results)

        print(data)

    def printDataset(self, dataset: PyDataset):
        for i, r in enumerate(dataset):
            if i > 10:
                break
            print(r)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        preprocessor = SequenceClassificationPreprocessor(
            model.model_dir, first_sequence='sentence', second_sequence=None)
        pipeline_ins = pipeline(
            task=Tasks.text_classification,
            model=model,
            preprocessor=preprocessor)
        self.predict(pipeline_ins)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        text_classification = pipeline(
            task=Tasks.text_classification, model=self.model_id)
        result = text_classification(
            PyDataset.load(
                'glue',
                subset_name='sst2',
                split='train',
                target='sentence',
                hub=Hubs.huggingface))
        self.printDataset(result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_default_model(self):
        text_classification = pipeline(task=Tasks.text_classification)
        result = text_classification(
            PyDataset.load(
                'glue',
                subset_name='sst2',
                split='train',
                target='sentence',
                hub=Hubs.huggingface))
        self.printDataset(result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_dataset(self):
        model = Model.from_pretrained(self.model_id)
        preprocessor = SequenceClassificationPreprocessor(
            model.model_dir, first_sequence='sentence', second_sequence=None)
        text_classification = pipeline(
            Tasks.text_classification, model=model, preprocessor=preprocessor)
        # loaded from huggingface dataset
        dataset = PyDataset.load(
            'glue',
            subset_name='sst2',
            split='train',
            target='sentence',
            hub=Hubs.huggingface)
        result = text_classification(dataset)
        self.printDataset(result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_modelscope_dataset(self):
        text_classification = pipeline(task=Tasks.text_classification)
        # loaded from modelscope dataset
        dataset = PyDataset.load(
            'squad', split='train', target='context', hub=Hubs.modelscope)
        result = text_classification(dataset)
        self.printDataset(result)


if __name__ == '__main__':
    unittest.main()
