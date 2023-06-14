# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import os
import shutil
import tempfile
import unittest

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import DownloadMode
from modelscope.utils.test_utils import test_level


class TestLoraDiffusionTrainer(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

        self.train_dataset = MsDataset.load(
            'buptwq/lora-stable-diffusion-finetune',
            split='train',
            download_mode=DownloadMode.FORCE_REDOWNLOAD)
        self.eval_dataset = MsDataset.load(
            'buptwq/lora-stable-diffusion-finetune',
            split='validation',
            download_mode=DownloadMode.FORCE_REDOWNLOAD)

        self.max_epochs = 100

        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_lora_diffusion_train(self):
        model_id = 'AI-ModelScope/stable-diffusion-v1-5'
        model_revision='v1.0.6'

        def cfg_modify_fn(cfg):
            cfg.train.max_epochs = self.max_epochs
            cfg.train.lr_scheduler = {
                'type': 'LambdaLR',
                'lr_lambda': lambda _: 1,
                'last_epoch': -1
            }
            cfg.train.optimizer.lr = 1e-4
            return cfg

        kwargs = dict(
            model=model_id,
            model_revision=model_revision,
            work_dir=self.tmp_dir,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            cfg_modify_fn=cfg_modify_fn)

        trainer = build_trainer(
            name=Trainers.lora_diffusion, default_args=kwargs)
        trainer.train()
        result = trainer.evaluate()
        print(f'Lora-diffusion train output: {result}.')

        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_lora_diffusion_eval(self):
        model_id = 'AI-ModelScope/stable-diffusion-v1-5'
        model_revision='v1.0.6'

        kwargs = dict(
            model=model_id,
            model_revision=model_revision,
            work_dir=self.tmp_dir,
            train_dataset=None,
            eval_dataset=self.eval_dataset)

        trainer = build_trainer(
            name=Trainers.lora_diffusion, default_args=kwargs)
        result = trainer.evaluate()
        print(f'Lora-diffusion eval output: {result}.')


if __name__ == '__main__':
    unittest.main()
