# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
import shutil
import tempfile
import unittest
from typing import Callable, List, Optional, Tuple, Union

import cv2
import torch
from torch.utils import data as data

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.models.cv.image_portrait_enhancement import \
    ImagePortraitEnhancement
from modelscope.msdatasets import MsDataset
from modelscope.msdatasets.task_datasets.image_portrait_enhancement import \
    ImagePortraitEnhancementDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import DownloadMode, ModelFile
from modelscope.utils.test_utils import test_level


class TestImagePortraitEnhancementTrainer(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        self.model_id = 'damo/cv_gpen_image-portrait-enhancement'

        dataset_train = MsDataset.load(
            'image-portrait-enhancement-dataset',
            namespace='modelscope',
            subset_name='default',
            split='test',
            download_mode=DownloadMode.FORCE_REDOWNLOAD)._hf_ds
        dataset_val = MsDataset.load(
            'image-portrait-enhancement-dataset',
            namespace='modelscope',
            subset_name='default',
            split='test',
            download_mode=DownloadMode.FORCE_REDOWNLOAD)._hf_ds

        self.dataset_train = ImagePortraitEnhancementDataset(
            dataset_train, is_train=True)
        self.dataset_val = ImagePortraitEnhancementDataset(
            dataset_val, is_train=False)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer(self):
        kwargs = dict(
            model=self.model_id,
            train_dataset=self.dataset_train,
            eval_dataset=self.dataset_val,
            device='gpu',
            max_epochs=1,
            work_dir=self.tmp_dir)

        trainer = build_trainer(
            name=Trainers.image_portrait_enhancement, default_args=kwargs)
        trainer.train()

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_trainer_with_model_and_args(self):
        tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        cache_path = snapshot_download(self.model_id)
        model = ImagePortraitEnhancement.from_pretrained(cache_path)
        kwargs = dict(
            cfg_file=os.path.join(cache_path, ModelFile.CONFIGURATION),
            model=model,
            train_dataset=self.dataset_train,
            eval_dataset=self.dataset_val,
            device='gpu',
            max_epochs=1,
            work_dir=self.tmp_dir)

        trainer = build_trainer(
            name=Trainers.image_portrait_enhancement, default_args=kwargs)
        trainer.train()


if __name__ == '__main__':
    unittest.main()
