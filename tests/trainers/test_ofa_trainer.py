# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import unittest

import json

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import DownloadMode, ModelFile
from modelscope.utils.hub import read_config
from modelscope.utils.test_utils import test_level


class TestOfaTrainer(unittest.TestCase):

    def setUp(self) -> None:
        self.finetune_cfg = \
            {'framework': 'pytorch',
             'task': 'ocr-recognition',
             'model': {'type': 'ofa',
                       'beam_search': {'beam_size': 5,
                                       'max_len_b': 64,
                                       'min_len': 1,
                                       'no_repeat_ngram_size': 0},
                       'seed': 7,
                       'max_src_length': 128,
                       'language': 'zh',
                       'gen_type': 'generation',
                       'patch_image_size': 480,
                       'is_document': False,
                       'max_image_size': 480,
                       'imagenet_default_mean_and_std': False},
             'pipeline': {'type': 'ofa-ocr-recognition'},
             'dataset': {'column_map': {'text': 'label'}},
             'train': {'work_dir': 'work/ckpts/recognition',
                       # 'launcher': 'pytorch',
                       'max_epochs': 1,
                       'use_fp16': True,
                       'dataloader': {'batch_size_per_gpu': 4, 'workers_per_gpu': 0},
                       'lr_scheduler': {'name': 'polynomial_decay',
                                        'warmup_proportion': 0.01,
                                        'lr_end': 1e-07},
                       'lr_scheduler_hook': {'type': 'LrSchedulerHook', 'by_epoch': False},
                       'optimizer': {'type': 'AdamW', 'lr': 5e-05, 'weight_decay': 0.01},
                       'optimizer_hook': {'type': 'TorchAMPOptimizerHook',
                                          'cumulative_iters': 1,
                                          'grad_clip': {'max_norm': 1.0, 'norm_type': 2},
                                          'loss_keys': 'loss'},
                       'criterion': {'name': 'AdjustLabelSmoothedCrossEntropyCriterion',
                                     'constraint_range': None,
                                     'drop_worst_after': 0,
                                     'drop_worst_ratio': 0.0,
                                     'ignore_eos': False,
                                     'ignore_prefix_size': 0,
                                     'label_smoothing': 0.1,
                                     'reg_alpha': 1.0,
                                     'report_accuracy': False,
                                     'sample_patch_num': 196,
                                     'sentence_avg': False,
                                     'use_rdrop': True},
                       'hooks': [{'type': 'BestCkptSaverHook',
                                  'metric_key': 'accuracy',
                                  'interval': 100},
                                 {'type': 'TextLoggerHook', 'interval': 1},
                                 {'type': 'IterTimerHook'},
                                 {'type': 'EvaluationHook', 'by_epoch': True, 'interval': 1}]},
             'evaluation': {'dataloader': {'batch_size_per_gpu': 4, 'workers_per_gpu': 0},
                            'metrics': [{'type': 'accuracy'}]},
             'preprocessor': []}

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer_std(self):
        WORKSPACE = './workspace/ckpts/recognition'
        os.makedirs(WORKSPACE, exist_ok=True)
        config_file = os.path.join(WORKSPACE, ModelFile.CONFIGURATION)
        with open(config_file, 'w') as writer:
            json.dump(self.finetune_cfg, writer, indent=4)
        pretrained_model = 'damo/ofa_ocr-recognition_scene_base_zh'

        args = dict(
            model=pretrained_model,
            work_dir=WORKSPACE,
            train_dataset=MsDataset.load(
                'ocr_fudanvi_zh',
                subset_name='scene',
                namespace='modelscope',
                split='train[800:900]',
                download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS),
            eval_dataset=MsDataset.load(
                'ocr_fudanvi_zh',
                subset_name='scene',
                namespace='modelscope',
                split='test[:20]',
                download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS),
            cfg_file=config_file)
        trainer = build_trainer(name=Trainers.ofa, default_args=args)
        trainer.train()

        self.assertIn(
            ModelFile.TORCH_MODEL_BIN_FILE,
            os.listdir(os.path.join(WORKSPACE, ModelFile.TRAIN_OUTPUT_DIR)))
        shutil.rmtree(WORKSPACE)


if __name__ == '__main__':
    unittest.main()
