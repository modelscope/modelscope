# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import json

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers.nlp.document_grounded_dialog_rerank_trainer import \
    DocumentGroundedDialogRerankTrainer
from modelscope.utils.config import Config
from modelscope.utils.constant import DownloadMode, ModelFile, Tasks
from modelscope.utils.test_utils import test_level


class TestDialogIntentTrainer(unittest.TestCase):

    def setUp(self):
        self.model_id = 'DAMO_ConvAI/nlp_convai_ranking_pretrain'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer_with_model_and_args(self):
        args = {
            'device': 'gpu',
            'tokenizer_name': '',
            'cache_dir': '',
            'instances_size': 1,
            'output_dir': './model',
            'max_num_seq_pairs_per_device': 32,
            'full_train_batch_size': 32,
            'gradient_accumulation_steps': 32,
            'per_gpu_train_batch_size': 1,
            'num_train_epochs': 1,
            'train_instances': -1,
            'learning_rate': 3e-5,
            'max_seq_length': 128,
            'num_labels': 2,
            'fold': '',  # IofN
            'doc_match_weight': 0.0,
            'query_length': 64,
            'resume_from': '',  # to resume training from a checkpoint
            'config_name': '',
            'do_lower_case': True,
            'weight_decay': 0.0,  # previous default was 0.01
            'adam_epsilon': 1e-8,
            'max_grad_norm': 1.0,
            'warmup_instances': 0,  # previous default was 0.1 of total
            'warmup_fraction': 0.0,  # only applies if warmup_instances <= 0
            'no_cuda': False,
            'n_gpu': 1,
            'seed': 42,
            'fp16': False,
            'fp16_opt_level': 'O1',  # previous default was O2
            'per_gpu_eval_batch_size': 8,
            'log_on_all_nodes': False,
            'world_size': 1,
            'global_rank': 0,
            'local_rank': -1,
            'tokenizer_resize': True,
            'model_resize': True
        }
        args[
            'gradient_accumulation_steps'] = args['full_train_batch_size'] // (
                args['per_gpu_train_batch_size'] * args['world_size'])
        data = MsDataset.load(
            'DAMO_ConvAI/FrDoc2BotRerank',
            download_mode=DownloadMode.FORCE_REDOWNLOAD,
            split='train')
        sub_train_dataset = [x for x in data][:10]
        trainer = DocumentGroundedDialogRerankTrainer(
            model=self.model_id, dataset=sub_train_dataset, args=args)
        trainer.train()


if __name__ == '__main__':
    unittest.main()
