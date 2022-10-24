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


class TestOfaTrainer(unittest.TestCase):

    def setUp(self) -> None:
        self.finetune_cfg = \
            {'framework': 'pytorch',
             'task': 'image-captioning',
             'model': {'type': 'ofa',
                       'beam_search': {'beam_size': 5,
                                       'max_len_b': 16,
                                       'min_len': 1,
                                       'no_repeat_ngram_size': 0},
                       'seed': 7,
                       'max_src_length': 256,
                       'language': 'en',
                       'gen_type': 'generation',
                       'patch_image_size': 480,
                       'max_image_size': 480,
                       'imagenet_default_mean_and_std': False},
             'pipeline': {'type': 'image-captioning'},
             'dataset': {'column_map': {'text': 'caption'}},
             'train': {'work_dir': 'work/ckpts/caption',
                       # 'launcher': 'pytorch',
                       'max_epochs': 1,
                       'use_fp16': True,
                       'dataloader': {'batch_size_per_gpu': 1, 'workers_per_gpu': 0},
                       'lr_scheduler': {'name': 'polynomial_decay',
                                        'warmup_proportion': 0.01,
                                        'lr_endo': 1e-07},
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
                                     'use_rdrop': False},
                       'hooks': [{'type': 'BestCkptSaverHook',
                                  'metric_key': 'bleu-4',
                                  'interval': 100},
                                 {'type': 'TextLoggerHook', 'interval': 1},
                                 {'type': 'IterTimerHook'},
                                 {'type': 'EvaluationHook', 'by_epoch': True, 'interval': 1}]},
             'evaluation': {'dataloader': {'batch_size_per_gpu': 4, 'workers_per_gpu': 0},
                            'metrics': [{'type': 'bleu',
                                         'eval_tokenized_bleu': False,
                                         'ref_name': 'labels',
                                         'hyp_name': 'caption'}]},
             'preprocessor': []}

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer_std(self):
        WORKSPACE = './workspace/ckpts/caption'
        os.makedirs(WORKSPACE, exist_ok=True)
        config_file = os.path.join(WORKSPACE, ModelFile.CONFIGURATION)
        with open(config_file, 'w') as writer:
            json.dump(self.finetune_cfg, writer)

        pretrained_model = 'damo/ofa_image-caption_coco_large_en'
        args = dict(
            model=pretrained_model,
            work_dir=WORKSPACE,
            train_dataset=MsDataset.load(
                'coco_2014_caption',
                namespace='modelscope',
                split='train[:20]'),
            eval_dataset=MsDataset.load(
                'coco_2014_caption',
                namespace='modelscope',
                split='validation[:10]'),
            metrics=[Metrics.BLEU],
            cfg_file=config_file)
        trainer = build_trainer(name=Trainers.ofa_tasks, default_args=args)
        trainer.train()

        self.assertIn(ModelFile.TORCH_MODEL_BIN_FILE,
                      os.path.join(WORKSPACE, 'output'))
        shutil.rmtree(WORKSPACE)


if __name__ == '__main__':
    unittest.main()
