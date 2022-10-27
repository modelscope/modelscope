# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.models.nlp import GPT3ForTextGeneration, PalmForTextGeneration
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import ModelFile
from modelscope.utils.test_utils import test_level


class TestFinetuneTextGeneration(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        from datasets import Dataset

        src_dataset_dict = {
            'src_txt': [
                'This is test sentence1-1', 'This is test sentence2-1',
                'This is test sentence3-1'
            ]
        }
        src_tgt_dataset_dict = {
            'src_txt':
            src_dataset_dict['src_txt'],
            'tgt_txt': [
                'This is test sentence1-2', 'This is test sentence2-2',
                'This is test sentence3-2'
            ]
        }

        self.src_dataset = MsDataset(Dataset.from_dict(src_dataset_dict))
        self.src_tgt_dataset = MsDataset(
            Dataset.from_dict(src_tgt_dataset_dict))

        self.max_epochs = 3

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer_with_palm(self):

        kwargs = dict(
            model='damo/nlp_palm2.0_text-generation_english-base',
            train_dataset=self.src_tgt_dataset,
            eval_dataset=self.src_tgt_dataset,
            max_epochs=self.max_epochs,
            work_dir=self.tmp_dir)

        trainer = build_trainer(
            name=Trainers.text_generation_trainer, default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_trainer_with_palm_with_model_and_args(self):

        cache_path = snapshot_download(
            'damo/nlp_palm2.0_text-generation_english-base')
        model = PalmForTextGeneration.from_pretrained(cache_path)
        kwargs = dict(
            cfg_file=os.path.join(cache_path, ModelFile.CONFIGURATION),
            model=model,
            train_dataset=self.src_tgt_dataset,
            eval_dataset=self.src_tgt_dataset,
            max_epochs=self.max_epochs,
            work_dir=self.tmp_dir)

        trainer = build_trainer(default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer_with_gpt3(self):

        kwargs = dict(
            model='damo/nlp_gpt3_text-generation_chinese-base',
            train_dataset=self.src_dataset,
            eval_dataset=self.src_dataset,
            max_epochs=self.max_epochs,
            work_dir=self.tmp_dir)

        trainer = build_trainer(
            name=Trainers.text_generation_trainer, default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_trainer_with_gpt3_with_model_and_args(self):

        cache_path = snapshot_download(
            'damo/nlp_gpt3_text-generation_chinese-base')
        model = GPT3ForTextGeneration.from_pretrained(cache_path)
        kwargs = dict(
            cfg_file=os.path.join(cache_path, ModelFile.CONFIGURATION),
            model=model,
            train_dataset=self.src_dataset,
            eval_dataset=self.src_dataset,
            max_epochs=self.max_epochs,
            work_dir=self.tmp_dir)

        trainer = build_trainer(default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

    @unittest.skip
    def test_finetune_cnndm(self):
        from modelscope.msdatasets import MsDataset
        dataset_dict = MsDataset.load('DuReader_robust-QG')
        train_dataset = dataset_dict['train'].to_hf_dataset() \
            .rename_columns({'text1': 'src_txt', 'text2': 'tgt_txt'})
        eval_dataset = dataset_dict['validation'].to_hf_dataset() \
            .rename_columns({'text1': 'src_txt', 'text2': 'tgt_txt'})
        num_warmup_steps = 200
        os.environ['LOCAL_RANK'] = '0'

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
            model='damo/nlp_palm2.0_text-generation_chinese-base',
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            work_dir=self.tmp_dir,
            cfg_modify_fn=cfg_modify_fn)
        trainer = build_trainer(
            name=Trainers.nlp_base_trainer, default_args=kwargs)
        trainer.train()


if __name__ == '__main__':
    unittest.main()
