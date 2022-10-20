# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

from modelscope.metainfo import Preprocessors, Trainers
from modelscope.models import Model
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.trainers import build_trainer
from modelscope.trainers.hooks import Hook
from modelscope.trainers.nlp_trainer import (EpochBasedTrainer,
                                             NlpEpochBasedTrainer)
from modelscope.trainers.optimizer.child_tuning_adamw_optimizer import \
    calculate_fisher
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.data_utils import to_device
from modelscope.utils.regress_test_utils import (MsRegressTool,
                                                 compare_arguments_nested)
from modelscope.utils.test_utils import test_level


class TestFinetuneSequenceClassification(unittest.TestCase):
    epoch_num = 1

    sentence1 = '今天气温比昨天高么？'
    sentence2 = '今天湿度比昨天高么？'

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.regress_tool = MsRegressTool(baseline=False)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_trainer_repeatable(self):
        import torch  # noqa

        def compare_fn(value1, value2, key, type):
            # Ignore the differences between optimizers of two torch versions
            if type != 'optimizer':
                return None

            match = (value1['type'] == value2['type'])
            shared_defaults = set(value1['defaults'].keys()).intersection(
                set(value2['defaults'].keys()))
            match = all([
                compare_arguments_nested(f'Optimizer defaults {key} not match',
                                         value1['defaults'][key],
                                         value2['defaults'][key])
                for key in shared_defaults
            ]) and match
            match = (len(value1['state_dict']['param_groups']) == len(
                value2['state_dict']['param_groups'])) and match
            for group1, group2 in zip(value1['state_dict']['param_groups'],
                                      value2['state_dict']['param_groups']):
                shared_keys = set(group1.keys()).intersection(
                    set(group2.keys()))
                match = all([
                    compare_arguments_nested(
                        f'Optimizer param_groups {key} not match', group1[key],
                        group2[key]) for key in shared_keys
                ]) and match
            return match

        def cfg_modify_fn(cfg):
            cfg.task = 'nli'
            cfg['preprocessor'] = {'type': 'nli-tokenizer'}
            cfg.train.optimizer.lr = 2e-5
            cfg['dataset'] = {
                'train': {
                    'labels': [
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                        '11', '12', '13', '14'
                    ],
                    'first_sequence':
                    'sentence',
                    'label':
                    'label',
                }
            }
            cfg.train.max_epochs = 5
            cfg.train.lr_scheduler = {
                'type': 'LinearLR',
                'start_factor': 1.0,
                'end_factor': 0.0,
                'total_iters':
                int(len(dataset['train']) / 32) * cfg.train.max_epochs,
                'options': {
                    'by_epoch': False
                }
            }
            cfg.train.hooks = [{
                'type': 'CheckpointHook',
                'interval': 1
            }, {
                'type': 'TextLoggerHook',
                'interval': 1
            }, {
                'type': 'IterTimerHook'
            }, {
                'type': 'EvaluationHook',
                'by_epoch': False,
                'interval': 100
            }]
            return cfg

        dataset = MsDataset.load('clue', subset_name='tnews')

        kwargs = dict(
            model='damo/nlp_structbert_backbone_base_std',
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            work_dir=self.tmp_dir,
            seed=42,
            cfg_modify_fn=cfg_modify_fn)

        os.environ['LOCAL_RANK'] = '0'
        trainer: EpochBasedTrainer = build_trainer(
            name=Trainers.nlp_base_trainer, default_args=kwargs)

        with self.regress_tool.monitor_ms_train(
                trainer, 'sbert-base-tnews', level='strict',
                compare_fn=compare_fn):
            trainer.train()

    def finetune(self,
                 model_id,
                 train_dataset,
                 eval_dataset,
                 name=Trainers.nlp_base_trainer,
                 cfg_modify_fn=None,
                 **kwargs):
        kwargs = dict(
            model=model_id,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            work_dir=self.tmp_dir,
            cfg_modify_fn=cfg_modify_fn,
            **kwargs)

        os.environ['LOCAL_RANK'] = '0'
        trainer = build_trainer(name=name, default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.epoch_num):
            self.assertIn(f'epoch_{i + 1}.pth', results_files)

        output_files = os.listdir(
            os.path.join(self.tmp_dir, ModelFile.TRAIN_OUTPUT_DIR))
        self.assertIn(ModelFile.CONFIGURATION, output_files)
        self.assertIn(ModelFile.TORCH_MODEL_BIN_FILE, output_files)
        copy_src_files = os.listdir(trainer.model_dir)

        print(f'copy_src_files are {copy_src_files}')
        print(f'output_files are {output_files}')
        for item in copy_src_files:
            if not item.startswith('.'):
                self.assertIn(item, output_files)

    def pipeline_sentence_similarity(self, model_dir):
        model = Model.from_pretrained(model_dir)
        pipeline_ins = pipeline(task=Tasks.sentence_similarity, model=model)
        print(pipeline_ins(input=(self.sentence1, self.sentence2)))

    @unittest.skip
    def test_finetune_afqmc(self):
        """This unittest is used to reproduce the clue:afqmc dataset + structbert model training results.

        User can train a custom dataset by modifying this piece of code and comment the @unittest.skip.
        """

        def cfg_modify_fn(cfg):
            cfg.task = Tasks.sentence_similarity
            cfg['preprocessor'] = {'type': Preprocessors.sen_sim_tokenizer}
            cfg.train.optimizer.lr = 2e-5
            cfg['dataset'] = {
                'train': {
                    'labels': ['0', '1'],
                    'first_sequence': 'sentence1',
                    'second_sequence': 'sentence2',
                    'label': 'label',
                }
            }
            cfg.train.max_epochs = self.epoch_num
            cfg.train.lr_scheduler = {
                'type': 'LinearLR',
                'start_factor': 1.0,
                'end_factor': 0.0,
                'total_iters':
                int(len(dataset['train']) / 32) * cfg.train.max_epochs,
                'options': {
                    'by_epoch': False
                }
            }
            cfg.train.hooks = [{
                'type': 'CheckpointHook',
                'interval': 1
            }, {
                'type': 'TextLoggerHook',
                'interval': 1
            }, {
                'type': 'IterTimerHook'
            }, {
                'type': 'EvaluationHook',
                'by_epoch': False,
                'interval': 100
            }]
            return cfg

        dataset = MsDataset.load('clue', subset_name='afqmc')
        self.finetune(
            model_id='damo/nlp_structbert_backbone_base_std',
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            cfg_modify_fn=cfg_modify_fn)

        output_dir = os.path.join(self.tmp_dir, ModelFile.TRAIN_OUTPUT_DIR)
        self.pipeline_sentence_similarity(output_dir)

    @unittest.skip
    def test_finetune_tnews(self):
        """This unittest is used to reproduce the clue:tnews dataset + structbert model training results.

        User can train a custom dataset by modifying this piece of code and comment the @unittest.skip.
        """

        def cfg_modify_fn(cfg):
            # TODO no proper task for tnews
            cfg.task = 'nli'
            cfg['preprocessor'] = {'type': 'nli-tokenizer'}
            cfg.train.optimizer.lr = 2e-5
            cfg['dataset'] = {
                'train': {
                    'labels': [
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                        '11', '12', '13', '14'
                    ],
                    'first_sequence':
                    'sentence',
                    'label':
                    'label',
                }
            }
            cfg.train.max_epochs = 5
            cfg.train.lr_scheduler = {
                'type': 'LinearLR',
                'start_factor': 1.0,
                'end_factor': 0.0,
                'total_iters':
                int(len(dataset['train']) / 32) * cfg.train.max_epochs,
                'options': {
                    'by_epoch': False
                }
            }
            cfg.train.hooks = [{
                'type': 'CheckpointHook',
                'interval': 1
            }, {
                'type': 'TextLoggerHook',
                'interval': 1
            }, {
                'type': 'IterTimerHook'
            }, {
                'type': 'EvaluationHook',
                'by_epoch': False,
                'interval': 100
            }]
            return cfg

        dataset = MsDataset.load('clue', subset_name='tnews')

        self.finetune(
            model_id='damo/nlp_structbert_backbone_base_std',
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            cfg_modify_fn=cfg_modify_fn)

    @unittest.skip
    def test_veco_xnli(self):
        """This unittest is used to reproduce the xnli dataset + veco model training results.

        Here we follow the training scenario listed in the Alicemind open source project:
        https://github.com/alibaba/AliceMind/tree/main/VECO
        by training the english language subset.
        User can train a custom dataset by modifying this piece of code and comment the @unittest.skip.
        """

        from datasets import load_dataset
        langs = ['en']
        langs_eval = ['en']
        train_datasets = []
        from datasets import DownloadConfig
        dc = DownloadConfig()
        dc.local_files_only = True
        for lang in langs:
            train_datasets.append(
                load_dataset('xnli', lang, split='train', download_config=dc))
        eval_datasets = []
        for lang in langs_eval:
            eval_datasets.append(
                load_dataset(
                    'xnli', lang, split='validation', download_config=dc))
        train_len = sum([len(dataset) for dataset in train_datasets])
        labels = ['0', '1', '2']

        def cfg_modify_fn(cfg):
            cfg.task = 'nli'
            cfg['preprocessor'] = {'type': 'nli-tokenizer'}
            cfg['dataset'] = {
                'train': {
                    'first_sequence': 'premise',
                    'second_sequence': 'hypothesis',
                    'labels': labels,
                    'label': 'label',
                }
            }
            cfg['train'] = {
                'work_dir':
                '/tmp',
                'max_epochs':
                2,
                'dataloader': {
                    'batch_size_per_gpu': 16,
                    'workers_per_gpu': 1
                },
                'optimizer': {
                    'type': 'AdamW',
                    'lr': 2e-5,
                    'options': {
                        'cumulative_iters': 8,
                    }
                },
                'lr_scheduler': {
                    'type': 'LinearLR',
                    'start_factor': 1.0,
                    'end_factor': 0.0,
                    'total_iters': int(train_len / 16) * 2,
                    'options': {
                        'by_epoch': False
                    }
                },
                'hooks': [{
                    'type': 'CheckpointHook',
                    'interval': 1,
                    'save_dir': '/root'
                }, {
                    'type': 'TextLoggerHook',
                    'interval': 1
                }, {
                    'type': 'IterTimerHook'
                }, {
                    'type': 'EvaluationHook',
                    'by_epoch': False,
                    'interval': 500
                }]
            }
            cfg['evaluation'] = {
                'dataloader': {
                    'batch_size_per_gpu': 128,
                    'workers_per_gpu': 1,
                    'shuffle': False
                }
            }
            return cfg

        self.finetune(
            'damo/nlp_veco_fill-mask-large',
            train_datasets,
            eval_datasets,
            name=Trainers.nlp_veco_trainer,
            cfg_modify_fn=cfg_modify_fn)

    @unittest.skip
    def test_finetune_cluewsc(self):
        """This unittest is used to reproduce the clue:wsc dataset + structbert model training results.

        A runnable sample of child-tuning is also showed here.

        User can train a custom dataset by modifying this piece of code and comment the @unittest.skip.
        """

        child_tuning_type = 'ChildTuning-F'
        mode = {}
        if child_tuning_type is not None:
            mode = {'mode': child_tuning_type, 'reserve_p': 0.2}

        def cfg_modify_fn(cfg):
            cfg.task = 'nli'
            cfg['preprocessor'] = {'type': 'nli-tokenizer'}
            cfg['dataset'] = {
                'train': {
                    'labels': ['0', '1'],
                    'first_sequence': 'text',
                    'second_sequence': 'text2',
                    'label': 'label',
                }
            }
            cfg.train.dataloader.batch_size_per_gpu = 16
            cfg.train.max_epochs = 30
            cfg.train.optimizer = {
                'type':
                'AdamW' if child_tuning_type is None else 'ChildTuningAdamW',
                'lr': 1e-5,
                'options': {},
                **mode,
            }
            cfg.train.lr_scheduler = {
                'type':
                'LinearLR',
                'start_factor':
                1.0,
                'end_factor':
                0.0,
                'total_iters':
                int(
                    len(dataset['train'])
                    / cfg.train.dataloader.batch_size_per_gpu)
                * cfg.train.max_epochs,
                'options': {
                    'by_epoch': False
                }
            }
            cfg.train.hooks = [{
                'type': 'CheckpointHook',
                'interval': 1
            }, {
                'type': 'TextLoggerHook',
                'interval': 1
            }, {
                'type': 'IterTimerHook'
            }, {
                'type': 'EvaluationHook',
                'by_epoch': False,
                'interval': 30
            }]
            return cfg

        def add_sentence2(features):
            return {
                'text2':
                features['target']['span2_text'] + '指代'
                + features['target']['span1_text']
            }

        dataset = MsDataset.load('clue', subset_name='cluewsc2020')
        dataset = {
            k: v.to_hf_dataset().map(add_sentence2)
            for k, v in dataset.items()
        }

        kwargs = dict(
            model='damo/nlp_structbert_backbone_base_std',
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            work_dir=self.tmp_dir,
            cfg_modify_fn=cfg_modify_fn)

        os.environ['LOCAL_RANK'] = '0'
        trainer: NlpEpochBasedTrainer = build_trainer(
            name=Trainers.nlp_base_trainer, default_args=kwargs)

        class CalculateFisherHook(Hook):

            @staticmethod
            def forward_step(model, inputs):
                inputs = to_device(inputs, trainer.device)
                trainer.train_step(model, inputs)
                return trainer.train_outputs['loss']

            def before_run(self, trainer: NlpEpochBasedTrainer):
                v = calculate_fisher(trainer.model, trainer.train_dataloader,
                                     self.forward_step, 0.2)
                trainer.optimizer.set_gradient_mask(v)

        if child_tuning_type == 'ChildTuning-D':
            trainer.register_hook(CalculateFisherHook())
        trainer.train()


if __name__ == '__main__':
    unittest.main()
