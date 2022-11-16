# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import os
import shutil
import tempfile
import unittest

import json
import torch

from modelscope.metainfo import Models, Pipelines, Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config
from modelscope.utils.constant import LogKeys, ModeKeys, Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import DistributedTestCase, test_level
from modelscope.utils.torch_utils import is_master


def train_func(work_dir, dist=False, log_interval=3, imgs_per_gpu=4):
    import easycv
    config_path = os.path.join(
        os.path.dirname(easycv.__file__),
        'configs/detection/yolox/yolox_s_8xb16_300e_coco.py')

    cfg = Config.from_file(config_path)

    cfg.log_config.update(
        dict(hooks=[
            dict(type='TextLoggerHook'),
            dict(type='TensorboardLoggerHook')
        ]))  # not support TensorboardLoggerHookV2

    ms_cfg_file = os.path.join(work_dir, 'ms_yolox_s_8xb16_300e_coco.json')
    from easycv.utils.ms_utils import to_ms_config

    if is_master():
        to_ms_config(
            cfg,
            dump=True,
            task=Tasks.image_object_detection,
            ms_model_name=Models.yolox,
            pipeline_name=Pipelines.easycv_detection,
            save_path=ms_cfg_file)

    trainer_name = Trainers.easycv
    train_dataset = MsDataset.load(
        dataset_name='small_coco_for_test', namespace='EasyCV', split='train')
    eval_dataset = MsDataset.load(
        dataset_name='small_coco_for_test',
        namespace='EasyCV',
        split='validation')

    cfg_options = {
        'train.max_epochs':
        2,
        'train.dataloader.batch_size_per_gpu':
        imgs_per_gpu,
        'evaluation.dataloader.batch_size_per_gpu':
        2,
        'train.hooks': [
            {
                'type': 'CheckpointHook',
                'interval': 1
            },
            {
                'type': 'EvaluationHook',
                'interval': 1
            },
            {
                'type': 'TextLoggerHook',
                'interval': log_interval
            },
        ]
    }
    kwargs = dict(
        cfg_file=ms_cfg_file,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        work_dir=work_dir,
        cfg_options=cfg_options,
        launcher='pytorch' if dist else None)

    trainer = build_trainer(trainer_name, kwargs)
    trainer.train()


@unittest.skipIf(not torch.cuda.is_available(), 'cuda unittest')
class EasyCVTrainerTestSingleGpu(unittest.TestCase):

    def setUp(self):
        self.logger = get_logger()
        self.logger.info(('Testing %s.%s' %
                          (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_single_gpu(self):
        train_func(self.tmp_dir)

        results_files = os.listdir(self.tmp_dir)
        json_files = glob.glob(os.path.join(self.tmp_dir, '*.log.json'))
        self.assertEqual(len(json_files), 1)

        with open(json_files[0], 'r', encoding='utf-8') as f:
            lines = [i.strip() for i in f.readlines()]

        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.TRAIN,
                LogKeys.EPOCH: 1,
                LogKeys.ITER: 3,
                LogKeys.LR: 0.00013
            }, json.loads(lines[0]))
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.EVAL,
                LogKeys.EPOCH: 1,
                LogKeys.ITER: 10
            }, json.loads(lines[1]))
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.TRAIN,
                LogKeys.EPOCH: 2,
                LogKeys.ITER: 3,
                LogKeys.LR: 0.00157
            }, json.loads(lines[2]))
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.EVAL,
                LogKeys.EPOCH: 2,
                LogKeys.ITER: 10
            }, json.loads(lines[3]))
        self.assertIn(f'{LogKeys.EPOCH}_1.pth', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_2.pth', results_files)
        for i in [0, 2]:
            self.assertIn(LogKeys.DATA_LOAD_TIME, lines[i])
            self.assertIn(LogKeys.ITER_TIME, lines[i])
            self.assertIn(LogKeys.MEMORY, lines[i])
            self.assertIn('total_loss', lines[i])
        for i in [1, 3]:
            self.assertIn(
                'CocoDetectionEvaluator_DetectionBoxes_Precision/mAP',
                lines[i])
            self.assertIn('DetectionBoxes_Precision/mAP', lines[i])
            self.assertIn('DetectionBoxes_Precision/mAP@.50IOU', lines[i])
            self.assertIn('DetectionBoxes_Precision/mAP@.75IOU', lines[i])
            self.assertIn('DetectionBoxes_Precision/mAP (small)', lines[i])


@unittest.skipIf(not torch.cuda.is_available()
                 or torch.cuda.device_count() <= 1, 'distributed unittest')
class EasyCVTrainerTestMultiGpus(DistributedTestCase):

    def setUp(self):
        self.logger = get_logger()
        self.logger.info(('Testing %s.%s' %
                          (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_multi_gpus(self):
        self.start(
            train_func,
            num_gpus=2,
            work_dir=self.tmp_dir,
            dist=True,
            log_interval=2,
            imgs_per_gpu=5)

        results_files = os.listdir(self.tmp_dir)
        json_files = glob.glob(os.path.join(self.tmp_dir, '*.log.json'))
        self.assertEqual(len(json_files), 1)

        with open(json_files[0], 'r', encoding='utf-8') as f:
            lines = [i.strip() for i in f.readlines()]

        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.TRAIN,
                LogKeys.EPOCH: 1,
                LogKeys.ITER: 2,
                LogKeys.LR: 0.0002
            }, json.loads(lines[0]))
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.EVAL,
                LogKeys.EPOCH: 1,
                LogKeys.ITER: 5
            }, json.loads(lines[1]))
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.TRAIN,
                LogKeys.EPOCH: 2,
                LogKeys.ITER: 2,
                LogKeys.LR: 0.0018
            }, json.loads(lines[2]))
        self.assertDictContainsSubset(
            {
                LogKeys.MODE: ModeKeys.EVAL,
                LogKeys.EPOCH: 2,
                LogKeys.ITER: 5
            }, json.loads(lines[3]))

        self.assertIn(f'{LogKeys.EPOCH}_1.pth', results_files)
        self.assertIn(f'{LogKeys.EPOCH}_2.pth', results_files)

        for i in [0, 2]:
            self.assertIn(LogKeys.DATA_LOAD_TIME, lines[i])
            self.assertIn(LogKeys.ITER_TIME, lines[i])
            self.assertIn(LogKeys.MEMORY, lines[i])
            self.assertIn('total_loss', lines[i])
        for i in [1, 3]:
            self.assertIn(
                'CocoDetectionEvaluator_DetectionBoxes_Precision/mAP',
                lines[i])
            self.assertIn('DetectionBoxes_Precision/mAP', lines[i])
            self.assertIn('DetectionBoxes_Precision/mAP@.50IOU', lines[i])
            self.assertIn('DetectionBoxes_Precision/mAP@.75IOU', lines[i])
            self.assertIn('DetectionBoxes_Precision/mAP (small)', lines[i])


if __name__ == '__main__':
    unittest.main()
