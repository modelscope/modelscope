# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import os
import shutil
import tempfile
import unittest

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.test_utils import test_level


class TestVisionEfficientTuningTrainer(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

        self.train_dataset = MsDataset.load(
            'foundation_model_evaluation_benchmark',
            namespace='damo',
            subset_name='OxfordFlowers',
            split='train')

        self.eval_dataset = MsDataset.load(
            'foundation_model_evaluation_benchmark',
            namespace='damo',
            subset_name='OxfordFlowers',
            split='eval')

        self.max_epochs = 1
        self.num_classes = 102
        self.tune_length = 10

        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_vision_efficient_tuning_adapter_train(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-adapter'

        def cfg_modify_fn(cfg):
            cfg.model.head.num_classes = self.num_classes
            cfg.model.finetune = True
            cfg.train.max_epochs = self.max_epochs
            cfg.train.lr_scheduler.T_max = self.max_epochs
            cfg.model.backbone.adapter_length = self.tune_length
            return cfg

        kwargs = dict(
            model=model_id,
            work_dir=self.tmp_dir,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            cfg_modify_fn=cfg_modify_fn)

        trainer = build_trainer(
            name=Trainers.vision_efficient_tuning, default_args=kwargs)
        trainer.train()
        result = trainer.evaluate()
        print(f'Vision-efficient-tuning-adapter train output: {result}.')

        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_vision_efficient_tuning_adapter_eval(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-adapter'

        kwargs = dict(
            model=model_id,
            work_dir=self.tmp_dir,
            train_dataset=None,
            eval_dataset=self.eval_dataset)

        trainer = build_trainer(
            name=Trainers.vision_efficient_tuning, default_args=kwargs)
        result = trainer.evaluate()
        print(f'Vision-efficient-tuning-adapter eval output: {result}.')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_vision_efficient_tuning_lora_train(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-lora'

        def cfg_modify_fn(cfg):
            cfg.model.head.num_classes = self.num_classes
            cfg.model.finetune = True
            cfg.train.max_epochs = self.max_epochs
            cfg.train.lr_scheduler.T_max = self.max_epochs
            cfg.model.backbone.lora_length = self.tune_length
            return cfg

        kwargs = dict(
            model=model_id,
            work_dir=self.tmp_dir,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            cfg_modify_fn=cfg_modify_fn)

        trainer = build_trainer(
            name=Trainers.vision_efficient_tuning, default_args=kwargs)
        trainer.train()
        result = trainer.evaluate()
        print(f'Vision-efficient-tuning-lora train output: {result}.')

        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_vision_efficient_tuning_lora_eval(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-lora'

        kwargs = dict(
            model=model_id,
            work_dir=self.tmp_dir,
            train_dataset=None,
            eval_dataset=self.eval_dataset)

        trainer = build_trainer(
            name=Trainers.vision_efficient_tuning, default_args=kwargs)
        result = trainer.evaluate()
        print(f'Vision-efficient-tuning-lora eval output: {result}.')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_vision_efficient_tuning_prefix_train(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-prefix'

        def cfg_modify_fn(cfg):
            cfg.model.head.num_classes = self.num_classes
            cfg.model.finetune = True
            cfg.train.max_epochs = self.max_epochs
            cfg.train.lr_scheduler.T_max = self.max_epochs
            cfg.model.backbone.prefix_length = self.tune_length
            return cfg

        kwargs = dict(
            model=model_id,
            work_dir=self.tmp_dir,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            cfg_modify_fn=cfg_modify_fn)

        trainer = build_trainer(
            name=Trainers.vision_efficient_tuning, default_args=kwargs)
        trainer.train()
        result = trainer.evaluate()
        print(f'Vision-efficient-tuning-prefix train output: {result}.')

        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_vision_efficient_tuning_prefix_eval(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-prefix'

        kwargs = dict(
            model=model_id,
            work_dir=self.tmp_dir,
            train_dataset=None,
            eval_dataset=self.eval_dataset)

        trainer = build_trainer(
            name=Trainers.vision_efficient_tuning, default_args=kwargs)
        result = trainer.evaluate()
        print(f'Vision-efficient-tuning-prefix eval output: {result}.')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_vision_efficient_tuning_prompt_train(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-prompt'

        def cfg_modify_fn(cfg):
            cfg.model.head.num_classes = self.num_classes
            cfg.model.finetune = True
            cfg.train.max_epochs = self.max_epochs
            cfg.train.lr_scheduler.T_max = self.max_epochs
            cfg.model.backbone.prompt_length = self.tune_length
            return cfg

        kwargs = dict(
            model=model_id,
            work_dir=self.tmp_dir,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            cfg_modify_fn=cfg_modify_fn)

        trainer = build_trainer(
            name=Trainers.vision_efficient_tuning, default_args=kwargs)
        trainer.train()
        result = trainer.evaluate()
        print(f'Vision-efficient-tuning-prompt train output: {result}.')

        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_vision_efficient_tuning_prompt_eval(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-prompt'

        kwargs = dict(
            model=model_id,
            work_dir=self.tmp_dir,
            train_dataset=None,
            eval_dataset=self.eval_dataset)

        trainer = build_trainer(
            name=Trainers.vision_efficient_tuning, default_args=kwargs)
        result = trainer.evaluate()
        print(f'Vision-efficient-tuning-prompt eval output: {result}.')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_vision_efficient_tuning_bitfit_train(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-bitfit'

        # model_id = '../modelcard/cv_vitb16_classification_vision-efficient-tuning-bitfit'
        def cfg_modify_fn(cfg):
            cfg.model.head.num_classes = self.num_classes
            cfg.model.finetune = True
            cfg.train.max_epochs = self.max_epochs
            cfg.train.lr_scheduler.T_max = self.max_epochs
            return cfg

        kwargs = dict(
            model=model_id,
            work_dir=self.tmp_dir,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            cfg_modify_fn=cfg_modify_fn)

        trainer = build_trainer(
            name=Trainers.vision_efficient_tuning, default_args=kwargs)
        trainer.train()
        result = trainer.evaluate()
        print(f'Vision-efficient-tuning-bitfit train output: {result}.')

        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_vision_efficient_tuning_bitfit_eval(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-bitfit'
        # model_id = '../modelcard/cv_vitb16_classification_vision-efficient-tuning-bitfit'
        kwargs = dict(
            model=model_id,
            work_dir=self.tmp_dir,
            train_dataset=None,
            eval_dataset=self.eval_dataset)

        trainer = build_trainer(
            name=Trainers.vision_efficient_tuning, default_args=kwargs)
        result = trainer.evaluate()
        print(f'Vision-efficient-tuning-bitfit eval output: {result}.')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_vision_efficient_tuning_sidetuning_train(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-sidetuning'

        def cfg_modify_fn(cfg):
            cfg.model.head.num_classes = self.num_classes
            cfg.model.finetune = True
            cfg.train.max_epochs = self.max_epochs
            cfg.train.lr_scheduler.T_max = self.max_epochs
            return cfg

        kwargs = dict(
            model=model_id,
            work_dir=self.tmp_dir,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            cfg_modify_fn=cfg_modify_fn)

        trainer = build_trainer(
            name=Trainers.vision_efficient_tuning, default_args=kwargs)
        trainer.train()
        result = trainer.evaluate()
        print(f'Vision-efficient-tuning-sidetuning train output: {result}.')

        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_vision_efficient_tuning_sidetuning_eval(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-sidetuning'
        kwargs = dict(
            model=model_id,
            work_dir=self.tmp_dir,
            train_dataset=None,
            eval_dataset=self.eval_dataset)

        trainer = build_trainer(
            name=Trainers.vision_efficient_tuning, default_args=kwargs)
        result = trainer.evaluate()
        print(f'Vision-efficient-tuning-sidetuning eval output: {result}.')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_vision_efficient_tuning_utuning_train(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-utuning'

        def cfg_modify_fn(cfg):
            cfg.model.head.num_classes = self.num_classes
            cfg.model.finetune = True
            cfg.train.max_epochs = self.max_epochs
            cfg.train.lr_scheduler.T_max = self.max_epochs
            return cfg

        kwargs = dict(
            model=model_id,
            work_dir=self.tmp_dir,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            cfg_modify_fn=cfg_modify_fn)

        trainer = build_trainer(
            name=Trainers.vision_efficient_tuning, default_args=kwargs)
        trainer.train()
        result = trainer.evaluate()
        print(f'Vision-efficient-tuning-utuning train output: {result}.')

        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_vision_efficient_tuning_utuning_eval(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-utuning'
        kwargs = dict(
            model=model_id,
            work_dir=self.tmp_dir,
            train_dataset=None,
            eval_dataset=self.eval_dataset)

        trainer = build_trainer(
            name=Trainers.vision_efficient_tuning, default_args=kwargs)
        result = trainer.evaluate()
        print(f'Vision-efficient-tuning-utuning eval output: {result}.')


if __name__ == '__main__':
    unittest.main()
