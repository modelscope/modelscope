# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import torch

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Preprocessors, Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import DownloadMode, ModelFile
from modelscope.utils.test_utils import test_level


class TestDialogModelingTrainer(unittest.TestCase):

    model_id = 'damo/nlp_space_pretrained-dialog-model'
    output_dir = './dialog_fintune_result'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer_with_model_and_args(self):
        # download data set
        data_multiwoz = MsDataset.load(
            'MultiWoz2.0', download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
        data_dir = os.path.join(
            data_multiwoz._hf_ds.config_kwargs['split_config']['train'],
            'data')

        # download model
        model_dir = snapshot_download(self.model_id)

        # dialog finetune config
        def cfg_modify_fn(cfg):
            config = {
                'seed': 10,
                'gpu': 1,
                'use_data_distributed': False,
                'valid_metric_name': '-loss',
                'num_epochs': 60,
                'save_dir': self.output_dir,
                'token_loss': True,
                'batch_size': 4,
                'log_steps': 10,
                'valid_steps': 0,
                'save_checkpoint': True,
                'save_summary': False,
                'shuffle': True,
                'sort_pool_size': 0
            }

            cfg.Trainer = config
            cfg.use_gpu = torch.cuda.is_available() and config['gpu'] >= 1
            return cfg

        # trainer config
        kwargs = dict(
            model_dir=model_dir,
            cfg_name='gen_train_config.json',
            data_dir=data_dir,
            cfg_modify_fn=cfg_modify_fn)

        trainer = build_trainer(
            name=Trainers.dialog_modeling_trainer, default_args=kwargs)
        assert trainer is not None

        # todo: it takes too long time to train and evaluate. It will be optimized later.
        """
        trainer.train()
        checkpoint_path = os.path.join(self.output_dir,
                                       ModelFile.TORCH_MODEL_BIN_FILE)
        assert os.path.exists(checkpoint_path)
        trainer.evaluate(checkpoint_path=checkpoint_path)
        """


if __name__ == '__main__':
    unittest.main()
