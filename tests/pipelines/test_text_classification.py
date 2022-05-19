# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
import tempfile
import unittest
import zipfile

from maas_lib.fileio import File
from maas_lib.models.nlp import SequenceClassificationModel
from maas_lib.pipelines import SequenceClassificationPipeline
from maas_lib.preprocessors import SequenceClassificationPreprocessor


class SequenceClassificationTest(unittest.TestCase):

    def predict(self, pipeline: SequenceClassificationPipeline):
        from easynlp.appzoo import load_dataset

        set = load_dataset('glue', 'sst2')
        data = set['test']['sentence'][:3]

        results = pipeline(data[0])
        print(results)
        results = pipeline(data[1])
        print(results)

        print(data)

    def test_run(self):
        model_url = 'https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com' \
                    '/release/easynlp_modelzoo/alibaba-pai/bert-base-sst2.zip'
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file = osp.join(tmp_dir, 'bert-base-sst2.zip')
            with open(tmp_file, 'wb') as ofile:
                ofile.write(File.read(model_url))
            with zipfile.ZipFile(tmp_file, 'r') as zipf:
                zipf.extractall(tmp_dir)
            path = osp.join(tmp_dir, 'bert-base-sst2')
            print(path)
            model = SequenceClassificationModel(path)
            preprocessor = SequenceClassificationPreprocessor(
                path, first_sequence='sentence', second_sequence=None)
            pipeline = SequenceClassificationPipeline(model, preprocessor)
            self.predict(pipeline)


if __name__ == '__main__':
    unittest.main()
