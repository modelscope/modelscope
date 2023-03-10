# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import DownloadMode
from modelscope.utils.test_utils import test_level


class TestActionDetectionTrainer(unittest.TestCase):

    def setUp(self):
        os.environ['OMP_NUM_THREADS'] = '1'
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        cmd_uninstall = ['pip', 'uninstall', '-y', 'detectron2']
        cmd = [
            'pip', 'install', '--upgrade',
            'git+https://gitee.com/lllcho/detectron2.git'
        ]
        subprocess.run(cmd_uninstall)
        subprocess.run(cmd)
        import detectron2
        print(f'Install detectron2 done, version {detectron2.__version__}')
        self.model_id = 'damo/cv_ResNetC3D_action-detection_detection2d'

        self.train_dataset = MsDataset.load(
            'lllcho/ivi_action',
            subset_name='default',
            split='train',
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_trainer(self):

        def cfg_modify_fn(cfg):
            cfg.train.max_iter = 5
            cfg.train.dataloader.batch_size_per_gpu = 1
            cfg.train.dataloader.workers_per_gpu = 1
            cfg.train.optimizer.lr = 1e-4
            cfg.train.lr_scheduler.warmup_step = 1
            cfg.train.checkpoint_interval = 5000

            cfg.evaluation.interval = 5000
            cfg.evaluation.dataloader.batch_size_per_gpu = 1
            cfg.evaluation.dataloader.workers_per_gpu = 1

            cfg.train.work_dir = self.tmp_dir
            cfg.train.num_gpus = 0
            return cfg

        trainer = build_trainer(
            Trainers.action_detection,
            dict(
                model_id=self.model_id,
                train_dataset=self.train_dataset,
                test_dataset=self.train_dataset,
                cfg_modify_fn=cfg_modify_fn))
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn('config.py', results_files)
        self.assertIn('model_final.pth', results_files)


if __name__ == '__main__':
    unittest.main()
