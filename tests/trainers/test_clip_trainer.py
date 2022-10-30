# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import unittest

import json

from modelscope.metainfo import Metrics, Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import ModelFile
from modelscope.utils.test_utils import test_level


class TestClipTrainer(unittest.TestCase):

    def setUp(self) -> None:
        self.finetune_cfg = \
            {'framework': 'pytorch',
             'task': 'multi-modal-embedding',
             'pipeline': {'type': 'multi-modal-embedding'},
             'pretrained_model': {'model_name': 'damo/multi-modal_clip-vit-base-patch16_zh'},
             'dataset': {'column_map': {'img': 'image', 'text': 'query'}},
             'train': {'work_dir': './workspace/ckpts/clip',
                       # 'launcher': 'pytorch',
                       'max_epochs': 1,
                       'use_fp16': True,
                       'dataloader': {'batch_size_per_gpu': 8,
                                      'workers_per_gpu': 0,
                                      'shuffle': True,
                                      'drop_last': True},
                       'lr_scheduler': {'name': 'cosine',
                                        'warmup_proportion': 0.01},
                       'lr_scheduler_hook': {'type': 'LrSchedulerHook', 'by_epoch': False},
                       'optimizer': {'type': 'AdamW'},
                       'optimizer_hparams': {'lr': 5e-05, 'weight_decay': 0.01},
                       'optimizer_hook': {'type': 'TorchAMPOptimizerHook',
                                          'cumulative_iters': 1,
                                          'loss_keys': 'loss'},
                       'loss_cfg': {'aggregate': True},
                       'hooks': [{'type': 'BestCkptSaverHook',
                                  'metric_key': 'inbatch_t2i_recall_at_1',
                                  'interval': 100},
                                 {'type': 'TextLoggerHook', 'interval': 1},
                                 {'type': 'IterTimerHook'},
                                 {'type': 'EvaluationHook', 'by_epoch': True, 'interval': 1},
                                 {'type': 'ClipClampLogitScaleHook'}]},
             'evaluation': {'dataloader': {'batch_size_per_gpu': 8,
                                           'workers_per_gpu': 0,
                                           'shuffle': True,
                                           'drop_last': True},
                            'metrics': [{'type': 'inbatch_recall'}]},
             'preprocessor': []}

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer_std(self):
        WORKSPACE = './workspace/ckpts/clip'
        os.makedirs(WORKSPACE, exist_ok=True)
        config_file = os.path.join(WORKSPACE, ModelFile.CONFIGURATION)
        with open(config_file, 'w') as writer:
            json.dump(self.finetune_cfg, writer)

        pretrained_model = 'damo/multi-modal_clip-vit-base-patch16_zh'
        args = dict(
            model=pretrained_model,
            work_dir=WORKSPACE,
            train_dataset=MsDataset.load(
                'muge', namespace='modelscope', split='train[:200]'),
            eval_dataset=MsDataset.load(
                'muge', namespace='modelscope', split='validation[:100]'),
            metrics=[Metrics.inbatch_recall],
            cfg_file=config_file)
        trainer = build_trainer(
            name=Trainers.clip_multi_modal_embedding, default_args=args)
        trainer.train()

        self.assertIn(ModelFile.TORCH_MODEL_BIN_FILE,
                      os.listdir(os.path.join(WORKSPACE, 'output')))
        shutil.rmtree(WORKSPACE)


if __name__ == '__main__':
    unittest.main()
