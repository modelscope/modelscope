# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from modelscope.metainfo import Trainers
from modelscope.models import Model
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.trainers import build_trainer
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.test_utils import test_level


class TestFinetuneSequenceClassification(unittest.TestCase):
    inputs = {
        'source_sentence': ["how long it take to get a master's degree"],
        'sentences_to_compare': [
            "On average, students take about 18 to 24 months to complete a master's degree.",
            'On the other hand, some students prefer to go at a slower pace and choose to take '
            'several years to complete their studies.',
            'It can take anywhere from two semesters'
        ]
    }

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
    def test_finetune_msmarco(self):

        def cfg_modify_fn(cfg):
            neg_sample = 4
            cfg.task = 'text-ranking'
            cfg['preprocessor'] = {'type': 'text-ranking'}
            cfg.train.optimizer.lr = 2e-5
            cfg['dataset'] = {
                'train': {
                    'type': 'bert',
                    'query_sequence': 'query',
                    'pos_sequence': 'positive_passages',
                    'neg_sequence': 'negative_passages',
                    'text_fileds': ['title', 'text'],
                    'qid_field': 'query_id',
                    'neg_sample': neg_sample
                },
                'val': {
                    'type': 'bert',
                    'query_sequence': 'query',
                    'pos_sequence': 'positive_passages',
                    'neg_sequence': 'negative_passages',
                    'text_fileds': ['title', 'text'],
                    'qid_field': 'query_id'
                },
            }
            cfg['evaluation']['dataloader']['batch_size_per_gpu'] = 30
            cfg.train.max_epochs = 1
            cfg.train.train_batch_size = 4
            cfg.train.lr_scheduler = {
                'type': 'LinearLR',
                'start_factor': 1.0,
                'end_factor': 0.0,
                'options': {
                    'by_epoch': False
                }
            }
            cfg.model['neg_sample'] = 4
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
                'interval': 15
            }]
            return cfg

        # load dataset
        ds = MsDataset.load('passage-ranking-demo', 'zyznull')
        train_ds = ds['train'].to_hf_dataset()
        dev_ds = ds['dev'].to_hf_dataset()

        model_id = 'damo/nlp_corom_passage-ranking_english-base'
        self.finetune(
            model_id=model_id,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            cfg_modify_fn=cfg_modify_fn)

        output_dir = os.path.join(self.tmp_dir, ModelFile.TRAIN_OUTPUT_DIR)
        self.pipeline_text_ranking(output_dir)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_finetune_dureader(self):

        def cfg_modify_fn(cfg):
            cfg.task = 'text-ranking'
            cfg['preprocessor'] = {'type': 'text-ranking'}
            cfg.train.optimizer.lr = 2e-5
            cfg['dataset'] = {
                'train': {
                    'type': 'bert',
                    'query_sequence': 'query',
                    'pos_sequence': 'positive_passages',
                    'neg_sequence': 'negative_passages',
                    'text_fileds': ['text'],
                    'qid_field': 'query_id'
                },
                'val': {
                    'type': 'bert',
                    'query_sequence': 'query',
                    'pos_sequence': 'positive_passages',
                    'neg_sequence': 'negative_passages',
                    'text_fileds': ['text'],
                    'qid_field': 'query_id'
                },
            }
            cfg['evaluation']['dataloader']['batch_size_per_gpu'] = 30
            cfg.train.max_epochs = 1
            cfg.train.train_batch_size = 4
            cfg.train.lr_scheduler = {
                'type': 'LinearLR',
                'start_factor': 1.0,
                'end_factor': 0.0,
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
                'interval': 5000
            }]
            return cfg

        # load dataset
        ds = MsDataset.load('dureader-retrieval-ranking', 'zyznull')
        train_ds = ds['train'].to_hf_dataset().shard(1000, index=0)
        dev_ds = ds['dev'].to_hf_dataset()
        model_id = 'damo/nlp_rom_passage-ranking_chinese-base'
        self.finetune(
            model_id=model_id,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            cfg_modify_fn=cfg_modify_fn)

    def pipeline_text_ranking(self, model_dir):
        model = Model.from_pretrained(model_dir)
        pipeline_ins = pipeline(task=Tasks.text_ranking, model=model)
        print(pipeline_ins(input=self.inputs))


if __name__ == '__main__':
    unittest.main()
