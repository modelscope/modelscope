# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from modelscope.metainfo import Preprocessors, Trainers
from modelscope.models import Model
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.trainers import build_trainer
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.test_utils import test_level


class TestFinetuneMGeo(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    def finetune(self,
                 model_id,
                 train_dataset,
                 eval_dataset,
                 name=Trainers.nlp_text_ranking_trainer,
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

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_finetune_geotes_rerank(self):

        def cfg_modify_fn(cfg):
            neg_sample = 19
            cfg.task = 'text-ranking'
            cfg['preprocessor'] = {'type': 'mgeo-ranking'}
            cfg.train.optimizer.lr = 5e-5
            cfg['dataset'] = {
                'train': {
                    'type': 'mgeo',
                    'query_sequence': 'query',
                    'pos_sequence': 'positive_passages',
                    'neg_sequence': 'negative_passages',
                    'text_fileds': ['text', 'gis'],
                    'qid_field': 'query_id',
                    'neg_sample': neg_sample,
                    'sequence_length': 64
                },
                'val': {
                    'type': 'mgeo',
                    'query_sequence': 'query',
                    'pos_sequence': 'positive_passages',
                    'neg_sequence': 'negative_passages',
                    'text_fileds': ['text', 'gis'],
                    'qid_field': 'query_id'
                },
            }
            cfg.evaluation.dataloader.batch_size_per_gpu = 16
            cfg.train.dataloader.batch_size_per_gpu = 3
            cfg.train.dataloader.workers_per_gpu = 16
            cfg.evaluation.dataloader.workers_per_gpu = 16
            cfg.train.train_iters_per_epoch = 10
            cfg.evaluation.val_iters_per_epoch = 10
            cfg['evaluation']['metrics'] = 'text-ranking-metric'
            cfg.train.max_epochs = 1
            cfg.model['neg_sample'] = neg_sample
            cfg.model['gis_num'] = 2
            cfg.model['finetune_mode'] = 'multi-modal'
            cfg.train.hooks = [{
                'type': 'CheckpointHook',
                'interval': 1
            }, {
                'type': 'TextLoggerHook',
                'interval': 100
            }, {
                'type': 'IterTimerHook'
            }, {
                'type': 'EvaluationHook',
                'by_epoch': True
            }]
            # lr_scheduler的配置

            cfg.train.lr_scheduler = {
                'type':
                'LinearLR',
                'start_factor':
                1.0,
                'end_factor':
                0.5,
                'total_iters':
                int(len(train_ds) / cfg.train.dataloader.batch_size_per_gpu)
                * cfg.train.max_epochs,
                'options': {
                    'warmup': {
                        'type':
                        'LinearWarmup',
                        'warmup_iters':
                        int(
                            len(train_ds)
                            / cfg.train.dataloader.batch_size_per_gpu)
                    },
                    'by_epoch': False
                }
            }

            return cfg

        # load dataset
        train_dataset = MsDataset.load(
            'GeoGLUE',
            subset_name='GeoTES-rerank',
            split='train',
            namespace='damo')
        dev_dataset = MsDataset.load(
            'GeoGLUE',
            subset_name='GeoTES-rerank',
            split='validation',
            namespace='damo')

        train_ds = train_dataset['train']
        dev_ds = dev_dataset['validation']

        model_id = 'damo/mgeo_backbone_chinese_base'
        self.finetune(
            model_id=model_id,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            cfg_modify_fn=cfg_modify_fn,
            name=Trainers.mgeo_ranking_trainer)

        output_dir = os.path.join(self.tmp_dir, ModelFile.TRAIN_OUTPUT_DIR)
        print(f'model is saved to {output_dir}')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_finetune_geoeag(self):

        def cfg_modify_fn(cfg):
            cfg.task = Tasks.sentence_similarity
            cfg['preprocessor'] = {'type': Preprocessors.sen_sim_tokenizer}

            cfg.train.dataloader.batch_size_per_gpu = 64
            cfg.evaluation.dataloader.batch_size_per_gpu = 64
            cfg.train.optimizer.lr = 2e-5
            cfg.train.max_epochs = 1
            cfg.train.train_iters_per_epoch = 10
            cfg.evaluation.val_iters_per_epoch = 10

            cfg['dataset'] = {
                'train': {
                    'labels': ['not_match', 'partial_match', 'exact_match'],
                    'first_sequence': 'sentence1',
                    'second_sequence': 'sentence2',
                    'label': 'label',
                    'sequence_length': 128
                }
            }
            cfg['evaluation']['metrics'] = 'seq-cls-metric'
            cfg.train.hooks = [{
                'type': 'CheckpointHook',
                'interval': 1
            }, {
                'type': 'TextLoggerHook',
                'interval': 100
            }, {
                'type': 'IterTimerHook'
            }, {
                'type': 'EvaluationHook',
                'by_epoch': True
            }]
            cfg.train.lr_scheduler.total_iters = int(
                len(train_dataset) / 32) * cfg.train.max_epochs
            return cfg

        # load dataset
        train_dataset = MsDataset.load(
            'GeoGLUE', subset_name='GeoEAG', split='train', namespace='damo')
        dev_dataset = MsDataset.load(
            'GeoGLUE',
            subset_name='GeoEAG',
            split='validation',
            namespace='damo')

        model_id = 'damo/mgeo_backbone_chinese_base'
        self.finetune(
            model_id=model_id,
            train_dataset=train_dataset['train'],
            eval_dataset=dev_dataset['validation'],
            cfg_modify_fn=cfg_modify_fn,
            name='nlp-base-trainer')

        output_dir = os.path.join(self.tmp_dir, ModelFile.TRAIN_OUTPUT_DIR)
        print(f'model is saved to {output_dir}')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_finetune_geoeta(self):

        def cfg_modify_fn(cfg):
            cfg.task = 'token-classification'
            cfg['dataset'] = {
                'train': {
                    'labels': label_enumerate_values,
                    'first_sequence': 'tokens',
                    'label': 'ner_tags',
                    'sequence_length': 128
                }
            }
            cfg['preprocessor'] = {
                'type': 'token-cls-tokenizer',
                'padding': 'max_length'
            }
            cfg.train.max_epochs = 1
            cfg.train.dataloader.batch_size_per_gpu = 32
            cfg.train.train_iters_per_epoch = 10
            cfg.evaluation.val_iters_per_epoch = 10
            cfg.train.optimizer.lr = 3e-5
            cfg.train.hooks = [{
                'type': 'CheckpointHook',
                'interval': 1
            }, {
                'type': 'TextLoggerHook',
                'interval': 100
            }, {
                'type': 'IterTimerHook'
            }, {
                'type': 'EvaluationHook',
                'by_epoch': True
            }]
            cfg.train.lr_scheduler.total_iters = int(
                len(train_dataset) / 32) * cfg.train.max_epochs

            return cfg

        def get_label_list(labels):
            unique_labels = set()
            for label in labels:
                unique_labels = unique_labels | set(label)
            label_list = list(unique_labels)
            label_list.sort()
            return label_list

        # load dataset
        train_dataset = MsDataset.load(
            'GeoGLUE', subset_name='GeoETA', split='train', namespace='damo')
        dev_dataset = MsDataset.load(
            'GeoGLUE',
            subset_name='GeoETA',
            split='validation',
            namespace='damo')

        label_enumerate_values = get_label_list(
            train_dataset._hf_ds['train']['ner_tags']
            + dev_dataset._hf_ds['validation']['ner_tags'])

        model_id = 'damo/mgeo_backbone_chinese_base'
        self.finetune(
            model_id=model_id,
            train_dataset=train_dataset['train'],
            eval_dataset=dev_dataset['validation'],
            cfg_modify_fn=cfg_modify_fn,
            name='nlp-base-trainer')

        output_dir = os.path.join(self.tmp_dir, ModelFile.TRAIN_OUTPUT_DIR)
        print(f'model is saved to {output_dir}')


if __name__ == '__main__':
    unittest.main()
