# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import shutil
import tempfile
import unittest

import json

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config
from modelscope.utils.constant import DownloadMode, ModelFile, Tasks
from modelscope.utils.test_utils import test_level


class TestDialogIntentTrainer(unittest.TestCase):

    def setUp(self):
        self.save_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def tearDown(self):
        shutil.rmtree(self.save_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer_with_model_and_args(self):
        model_id = 'damo/nlp_space_pretrained-dialog-model'
        data_banking = MsDataset.load('banking77')
        self.data_dir = data_banking._hf_ds.config_kwargs['split_config'][
            'train']
        self.model_dir = snapshot_download(model_id)
        self.debugging = True
        kwargs = dict(
            model_dir=self.model_dir,
            cfg_name='intent_train_config.json',
            cfg_modify_fn=self.cfg_modify_fn)
        trainer = build_trainer(
            name=Trainers.dialog_intent_trainer, default_args=kwargs)
        trainer.train()

    def cfg_modify_fn(self, cfg):
        config = {
            'num_intent': 77,
            'BPETextField': {
                'vocab_path': '',
                'data_name': 'banking77',
                'data_root': self.data_dir,
                'understand': True,
                'generation': False,
                'max_len': 256
            },
            'Dataset': {
                'data_dir': self.data_dir,
                'with_contrastive': False,
                'trigger_role': 'user',
                'trigger_data': 'banking'
            },
            'Trainer': {
                'can_norm': True,
                'seed': 11,
                'gpu': 1,
                'save_dir': self.save_dir,
                'batch_size_label': 128,
                'batch_size_nolabel': 0,
                'log_steps': 20
            },
            'Model': {
                'init_checkpoint': self.model_dir,
                'model': 'IntentUnifiedTransformer',
                'example': False,
                'num_intent': 77,
                'with_rdrop': True,
                'num_turn_embeddings': 21,
                'dropout': 0.25,
                'kl_ratio': 5.0,
                'embed_dropout': 0.25,
                'attn_dropout': 0.25,
                'ff_dropout': 0.25,
                'with_pool': False,
                'warmup_steps': -1
            }
        }
        cfg.BPETextField.vocab_path = os.path.join(self.model_dir,
                                                   ModelFile.VOCAB_FILE)
        cfg.num_intent = 77
        cfg.Trainer.update(config['Trainer'])
        cfg.BPETextField.update(config['BPETextField'])
        cfg.Dataset.update(config['Dataset'])
        cfg.Model.update(config['Model'])
        if self.debugging:
            cfg.Trainer.save_checkpoint = False
            cfg.Trainer.num_epochs = 1
            cfg.Trainer.batch_size_label = 64
        return cfg


if __name__ == '__main__':
    unittest.main()
