# Copyright (c) Alibaba, Inc. and its affiliates.
import shutil
import unittest
import zipfile
from pathlib import Path

from maas_lib.fileio import File
from maas_lib.models import Model
from maas_lib.models.nlp import BertForSequenceClassification
from maas_lib.pipelines import SequenceClassificationPipeline, pipeline, util
from maas_lib.preprocessors import SequenceClassificationPreprocessor
from maas_lib.pydatasets import PyDataset
from maas_lib.utils.constant import Tasks


class SequenceClassificationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_id = 'damo/bert-base-sst2'
        # switch to False if downloading everytime is not desired
        purge_cache = True
        if purge_cache:
            shutil.rmtree(
                util.get_model_cache_dir(self.model_id), ignore_errors=True)

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

    def test_run(self):
        model_url = 'https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com' \
                    '/release/easynlp_modelzoo/alibaba-pai/bert-base-sst2.zip'
        cache_path_str = r'.cache/easynlp/bert-base-sst2.zip'
        cache_path = Path(cache_path_str)

        if not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.touch(exist_ok=True)
            with cache_path.open('wb') as ofile:
                ofile.write(File.read(model_url))

        with zipfile.ZipFile(cache_path_str, 'r') as zipf:
            zipf.extractall(cache_path.parent)
        path = r'.cache/easynlp/'
        model = BertForSequenceClassification(path)
        preprocessor = SequenceClassificationPreprocessor(
            path, first_sequence='sentence', second_sequence=None)
        pipeline1 = SequenceClassificationPipeline(model, preprocessor)
        self.predict(pipeline1)
        pipeline2 = pipeline(
            Tasks.text_classification, model=model, preprocessor=preprocessor)
        print(pipeline2('Hello world!'))

    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        preprocessor = SequenceClassificationPreprocessor(
            model.model_dir, first_sequence='sentence', second_sequence=None)
        pipeline_ins = pipeline(
            task=Tasks.text_classification,
            model=model,
            preprocessor=preprocessor)
        self.predict(pipeline_ins)

    def test_run_with_model_name(self):
        text_classification = pipeline(
            task=Tasks.text_classification, model=self.model_id)
        result = text_classification(
            PyDataset.load('glue', name='sst2', target='sentence'))
        self.printDataset(result)

    def test_run_with_dataset(self):
        model = Model.from_pretrained(self.model_id)
        preprocessor = SequenceClassificationPreprocessor(
            model.model_dir, first_sequence='sentence', second_sequence=None)
        text_classification = pipeline(
            Tasks.text_classification, model=model, preprocessor=preprocessor)
        # loaded from huggingface dataset
        # TODO: add load_from parameter (an enum) LOAD_FROM.hugging_face
        # TODO: rename parameter as dataset_name and subset_name
        dataset = PyDataset.load('glue', name='sst2', target='sentence')
        result = text_classification(dataset)
        self.printDataset(result)


if __name__ == '__main__':
    unittest.main()
