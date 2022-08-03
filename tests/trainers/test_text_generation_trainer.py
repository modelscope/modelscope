# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models.nlp.palm_v2 import PalmForTextGeneration
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import ModelFile
from modelscope.utils.test_utils import test_level


class TestTextGenerationTrainer(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        self.model_id = 'damo/nlp_palm2.0_text-generation_english-base'

        # todo: Replace below scripts with MsDataset.load when the formal dataset service is ready
        from datasets import Dataset
        dataset_dict = {
            'src_txt': [
                'This is test sentence1-1', 'This is test sentence2-1',
                'This is test sentence3-1'
            ],
            'tgt_txt': [
                'This is test sentence1-2', 'This is test sentence2-2',
                'This is test sentence3-2'
            ]
        }
        dataset = Dataset.from_dict(dataset_dict)

        class MsDatasetDummy(MsDataset):

            def __len__(self):
                return len(self._hf_ds)

        self.dataset = MsDatasetDummy(dataset)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer(self):

        def cfg_modify_fn(cfg):
            cfg.preprocessor.type = 'text-gen-tokenizer'
            return cfg

        kwargs = dict(
            model=self.model_id,
            train_dataset=self.dataset,
            eval_dataset=self.dataset,
            work_dir=self.tmp_dir,
            cfg_modify_fn=cfg_modify_fn,
            model_revision='beta')

        trainer = build_trainer(
            name='NlpEpochBasedTrainer', default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(3):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_trainer_with_model_and_args(self):
        tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        cache_path = snapshot_download(self.model_id, revision='beta')
        model = PalmForTextGeneration.from_pretrained(cache_path)
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

    @unittest.skip
    def test_finetune_cnndm(self):
        from datasets import load_dataset
        dataset_dict = load_dataset('ccdv/cnn_dailymail', '3.0.0')
        train_dataset = dataset_dict['train'] \
            .rename_columns({'article': 'src_txt', 'highlights': 'tgt_txt'}) \
            .remove_columns('id')
        eval_dataset = dataset_dict['validation'] \
            .rename_columns({'article': 'src_txt', 'highlights': 'tgt_txt'}) \
            .remove_columns('id')
        num_warmup_steps = 2000

        def noam_lambda(current_step: int):
            current_step += 1
            return min(current_step**(-0.5),
                       current_step * num_warmup_steps**(-1.5))

        def cfg_modify_fn(cfg):
            cfg.train.lr_scheduler = {
                'type': 'LambdaLR',
                'lr_lambda': noam_lambda,
                'options': {
                    'by_epoch': False
                }
            }
            return cfg

        kwargs = dict(
            model=self.model_id,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            work_dir=self.tmp_dir,
            cfg_modify_fn=cfg_modify_fn,
            model_revision='beta')
        trainer = build_trainer(
            name='NlpEpochBasedTrainer', default_args=kwargs)
        trainer.train()


if __name__ == '__main__':
    unittest.main()
