# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models.nlp.sbert_for_sequence_classification import \
    SbertTextClassfier
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import ModelFile
from modelscope.utils.test_utils import test_level


class TestTrainerWithNlp(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        from datasets import Dataset
        dataset_dict = {
            'sentence1': [
                'This is test sentence1-1', 'This is test sentence2-1',
                'This is test sentence3-1'
            ],
            'sentence2': [
                'This is test sentence1-2', 'This is test sentence2-2',
                'This is test sentence3-2'
            ],
            'label': [0, 1, 1]
        }
        dataset = Dataset.from_dict(dataset_dict)
        self.dataset = MsDataset.from_hf_dataset(dataset)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_trainer(self):
        model_id = 'damo/nlp_structbert_sentence-similarity_chinese-base'
        kwargs = dict(
            model=model_id,
            train_dataset=self.dataset,
            eval_dataset=self.dataset,
            work_dir=self.tmp_dir)

        trainer = build_trainer(default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(10):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_trainer_with_model_and_args(self):
        tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        model_id = 'damo/nlp_structbert_sentence-similarity_chinese-base'
        cache_path = snapshot_download(model_id)
        model = SbertTextClassfier.from_pretrained(cache_path)
        kwargs = dict(
            cfg_file=os.path.join(cache_path, ModelFile.CONFIGURATION),
            model=model,
            train_dataset=self.dataset,
            eval_dataset=self.dataset,
            max_epochs=2,
            work_dir=self.tmp_dir)

        trainer = build_trainer(default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(2):
            self.assertIn(f'epoch_{i+1}.pth', results_files)


if __name__ == '__main__':
    unittest.main()
