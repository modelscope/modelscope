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


class TestEfficientDiffusionTuningTrainer(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

        self.train_dataset = MsDataset.load(
            'controlnet_dataset_condition_fill50k',
            namespace='damo',
            split='train',
            download_mode=DownloadMode.FORCE_REDOWNLOAD).to_hf_dataset(  # noqa
            ).select(range(100))  # noqa
        self.eval_dataset = MsDataset.load(
            'controlnet_dataset_condition_fill50k',
            namespace='damo',
            split='validation',
            download_mode=DownloadMode.FORCE_REDOWNLOAD).to_hf_dataset(  # noqa
            ).select(range(20))  # noqa

        self.max_epochs = 1

        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_efficient_diffusion_tuning_lora_train(self):
        model_id = 'damo/multi-modal_efficient-diffusion-tuning-lora'

        def cfg_modify_fn(cfg):
            cfg.train.max_epochs = self.max_epochs
            cfg.train.lr_scheduler.T_max = self.max_epochs
            cfg.model.inference = False
            return cfg

        kwargs = dict(
            model=model_id,
            work_dir=self.tmp_dir,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            cfg_modify_fn=cfg_modify_fn)

        trainer = build_trainer(
            name=Trainers.efficient_diffusion_tuning, default_args=kwargs)
        trainer.train()
        result = trainer.evaluate()
        print(f'Efficient-diffusion-tuning-lora train output: {result}.')

        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_efficient_diffusion_tuning_lora_eval(self):
        model_id = 'damo/multi-modal_efficient-diffusion-tuning-lora'

        def cfg_modify_fn(cfg):
            cfg.model.inference = False
            return cfg

        kwargs = dict(
            model=model_id,
            work_dir=self.tmp_dir,
            train_dataset=None,
            eval_dataset=self.eval_dataset,
            cfg_modify_fn=cfg_modify_fn)

        trainer = build_trainer(
            name=Trainers.efficient_diffusion_tuning, default_args=kwargs)
        result = trainer.evaluate()
        print(f'Efficient-diffusion-tuning-lora eval output: {result}.')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_efficient_diffusion_tuning_control_lora_train(self):
        model_id = 'damo/multi-modal_efficient-diffusion-tuning-control-lora'

        def cfg_modify_fn(cfg):
            cfg.train.max_epochs = self.max_epochs
            cfg.train.lr_scheduler.T_max = self.max_epochs
            cfg.model.inference = False
            return cfg

        kwargs = dict(
            model=model_id,
            work_dir=self.tmp_dir,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            cfg_modify_fn=cfg_modify_fn)

        trainer = build_trainer(
            name=Trainers.efficient_diffusion_tuning, default_args=kwargs)
        trainer.train()
        result = trainer.evaluate()
        print(
            f'Efficient-diffusion-tuning-control-lora train output: {result}.')

        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_efficient_diffusion_tuning_control_lora_eval(self):
        model_id = 'damo/multi-modal_efficient-diffusion-tuning-control-lora'

        def cfg_modify_fn(cfg):
            cfg.model.inference = False
            return cfg

        kwargs = dict(
            model=model_id,
            work_dir=self.tmp_dir,
            train_dataset=None,
            eval_dataset=self.eval_dataset,
            cfg_modify_fn=cfg_modify_fn)

        trainer = build_trainer(
            name=Trainers.efficient_diffusion_tuning, default_args=kwargs)
        result = trainer.evaluate()
        print(
            f'Efficient-diffusion-tuning-control-lora eval output: {result}.')


if __name__ == '__main__':
    unittest.main()
