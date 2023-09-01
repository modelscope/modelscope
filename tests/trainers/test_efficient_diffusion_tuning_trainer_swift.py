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


class TestEfficientDiffusionTuningTrainerSwift(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

        self.train_dataset = MsDataset.load(
            'style_custom_dataset',
            namespace='damo',
            split='train',
            subset_name='Anime').remap_columns({'Image:FILE': 'target:FILE'})

        self.max_epochs = 30
        self.lr = 0.0001

        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_efficient_diffusion_tuning_swift_lora_train(self):
        model_id = 'damo/multi-modal_efficient-diffusion-tuning-swift-lora'
        model_revision = 'v1.0.2'

        def cfg_modify_fn(cfg):
            cfg.train.max_epochs = self.max_epochs
            cfg.train.lr_scheduler.T_max = self.max_epochs
            cfg.train.optimizer.lr = self.lr
            cfg.model.inference = False
            cfg.model.pretrained_tuner = None
            return cfg

        kwargs = dict(
            model=model_id,
            model_revision=model_revision,
            work_dir=self.tmp_dir,
            train_dataset=self.train_dataset,
            cfg_modify_fn=cfg_modify_fn)

        trainer = build_trainer(
            name=Trainers.efficient_diffusion_tuning, default_args=kwargs)
        trainer.train()
        print('Efficient-diffusion-tuning-swift-lora train.')

        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        self.assertIn(f'epoch_{self.max_epochs}.pth', results_files)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_efficient_diffusion_tuning_swift_adapter_train(self):
        model_id = 'damo/multi-modal_efficient-diffusion-tuning-swift-adapter'
        model_revision = 'v1.0.2'

        def cfg_modify_fn(cfg):
            cfg.train.max_epochs = self.max_epochs
            cfg.train.lr_scheduler.T_max = self.max_epochs
            cfg.train.optimizer.lr = self.lr
            cfg.model.inference = False
            cfg.model.pretrained_tuner = None
            return cfg

        kwargs = dict(
            model=model_id,
            model_revision=model_revision,
            work_dir=self.tmp_dir,
            train_dataset=self.train_dataset,
            cfg_modify_fn=cfg_modify_fn)

        trainer = build_trainer(
            name=Trainers.efficient_diffusion_tuning, default_args=kwargs)
        trainer.train()
        print('Efficient-diffusion-tuning-swift-adapter train.')

        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        self.assertIn(f'epoch_{self.max_epochs}.pth', results_files)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_efficient_diffusion_tuning_swift_prompt_train(self):
        model_id = 'damo/multi-modal_efficient-diffusion-tuning-swift-prompt'
        model_revision = 'v1.0.2'

        def cfg_modify_fn(cfg):
            cfg.train.max_epochs = self.max_epochs
            cfg.train.lr_scheduler.T_max = self.max_epochs
            cfg.train.optimizer.lr = self.lr
            cfg.model.inference = False
            cfg.model.pretrained_tuner = None
            return cfg

        kwargs = dict(
            model=model_id,
            model_revision=model_revision,
            work_dir=self.tmp_dir,
            train_dataset=self.train_dataset,
            cfg_modify_fn=cfg_modify_fn)

        trainer = build_trainer(
            name=Trainers.efficient_diffusion_tuning, default_args=kwargs)
        trainer.train()
        print('Efficient-diffusion-tuning-swift-prompt train.')
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        self.assertIn(f'epoch_{self.max_epochs}.pth', results_files)


if __name__ == '__main__':
    unittest.main()
