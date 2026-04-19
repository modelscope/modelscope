# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import os
import shutil
import tempfile
import unittest

import torch

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile
from modelscope.utils.import_utils import exists
from modelscope.utils.test_utils import DistributedTestCase, test_level


def _setup():
    model_id = 'damo/cv_resnet_carddetection_scrfd34gkps'
    # mini dataset only for unit test, remove '_mini' for full dataset.
    ms_ds_syncards = MsDataset.load(
        'SyntheticCards_mini', namespace='shaoxuan')

    data_path = ms_ds_syncards.config_kwargs['split_config']
    train_dir = data_path['train']
    val_dir = data_path['validation']
    train_root = train_dir + '/' + os.listdir(train_dir)[0] + '/'
    val_root = val_dir + '/' + os.listdir(val_dir)[0] + '/'
    max_epochs = 1  # run epochs in unit test

    cache_path = snapshot_download(model_id)

    tmp_dir = tempfile.TemporaryDirectory().name
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    return train_root, val_root, max_epochs, cache_path, tmp_dir


def train_func(**kwargs):
    trainer = build_trainer(
        name=Trainers.card_detection_scrfd, default_args=kwargs)
    trainer.train()


class TestCardDetectionScrfdTrainerSingleGPU(unittest.TestCase):

    def setUp(self):
        print(('SingleGPU Testing %s.%s' %
               (type(self).__name__, self._testMethodName)))
        self.train_root, self.val_root, self.max_epochs, self.cache_path, self.tmp_dir = _setup(
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    def _cfg_modify_fn(self, cfg):
        cfg.checkpoint_config.interval = 1
        cfg.log_config.interval = 10
        cfg.evaluation.interval = 1
        cfg.data.workers_per_gpu = 3
        cfg.data.samples_per_gpu = 4  # batch size
        return cfg

    @unittest.skipUnless(
        exists('transformers<5.0'),
        'Skip test because transformers version is too high.')
    def test_trainer_from_scratch(self):
        kwargs = dict(
            cfg_file=os.path.join(self.cache_path, 'mmcv_scrfd.py'),
            work_dir=self.tmp_dir,
            train_root=self.train_root,
            val_root=self.val_root,
            total_epochs=self.max_epochs,
            cfg_modify_fn=self._cfg_modify_fn)

        trainer = build_trainer(
            name=Trainers.card_detection_scrfd, default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_trainer_finetune(self):
        pretrain_epoch = 640
        self.max_epochs += pretrain_epoch
        kwargs = dict(
            cfg_file=os.path.join(self.cache_path, 'mmcv_scrfd.py'),
            work_dir=self.tmp_dir,
            train_root=self.train_root,
            val_root=self.val_root,
            total_epochs=self.max_epochs,
            resume_from=os.path.join(self.cache_path,
                                     ModelFile.TORCH_MODEL_BIN_FILE),
            cfg_modify_fn=self._cfg_modify_fn)

        trainer = build_trainer(
            name=Trainers.card_detection_scrfd, default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(pretrain_epoch, self.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)


@unittest.skipIf(not torch.cuda.is_available()
                 or torch.cuda.device_count() <= 1, 'distributed unittest')
class TestCardDetectionScrfdTrainerMultiGpus(DistributedTestCase):

    def setUp(self):
        print(('MultiGPUs Testing %s.%s' %
               (type(self).__name__, self._testMethodName)))
        self.train_root, self.val_root, self.max_epochs, self.cache_path, self.tmp_dir = _setup(
        )
        cfg_file_path = os.path.join(self.cache_path, 'mmcv_scrfd.py')
        cfg = Config.from_file(cfg_file_path)
        cfg.checkpoint_config.interval = 1
        cfg.log_config.interval = 10
        cfg.evaluation.interval = 1
        cfg.data.workers_per_gpu = 3
        cfg.data.samples_per_gpu = 4
        cfg.dump(cfg_file_path)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_multi_gpus_finetune(self):
        pretrain_epoch = 640
        self.max_epochs += pretrain_epoch
        kwargs = dict(
            cfg_file=os.path.join(self.cache_path, 'mmcv_scrfd.py'),
            work_dir=self.tmp_dir,
            train_root=self.train_root,
            val_root=self.val_root,
            total_epochs=self.max_epochs,
            resume_from=os.path.join(self.cache_path,
                                     ModelFile.TORCH_MODEL_BIN_FILE),
            launcher='pytorch')
        self.start(train_func, num_gpus=2, **kwargs)
        results_files = os.listdir(self.tmp_dir)
        json_files = glob.glob(os.path.join(self.tmp_dir, '*.log.json'))
        self.assertEqual(len(json_files), 1)
        for i in range(pretrain_epoch, self.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)


if __name__ == '__main__':
    unittest.main()
