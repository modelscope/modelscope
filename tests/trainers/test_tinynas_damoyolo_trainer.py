# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import os
import shutil
import tempfile
import unittest

import torch

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile
from modelscope.utils.test_utils import DistributedTestCase, test_level


def _setup():
    model_id = 'damo/cv_tinynas_object-detection_damoyolo'
    cache_path = snapshot_download(model_id)
    return cache_path


class TestTinynasDamoyoloTrainerSingleGPU(unittest.TestCase):

    def setUp(self):
        self.model_id = 'damo/cv_tinynas_object-detection_damoyolo'
        self.cache_path = _setup()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer_from_scratch_singleGPU(self):
        kwargs = dict(
            cfg_file=os.path.join(self.cache_path, 'configuration.json'),
            gpu_ids=[
                0,
            ],
            batch_size=2,
            max_epochs=3,
            num_classes=80,
            base_lr_per_img=0.001,
            cache_path=self.cache_path,
            train_image_dir='./data/test/images/image_detection/images',
            val_image_dir='./data/test/images/image_detection/images',
            train_ann=
            './data/test/images/image_detection/annotations/coco_sample.json',
            val_ann=
            './data/test/images/image_detection/annotations/coco_sample.json',
        )
        trainer = build_trainer(
            name=Trainers.tinynas_damoyolo, default_args=kwargs)
        trainer.train()
        trainer.evaluate(
            checkpoint_path=os.path.join('./workdirs/damoyolo_s',
                                         'epoch_3_ckpt.pth'))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer_from_scratch_singleGPU_model_id(self):
        kwargs = dict(
            model=self.model_id,
            gpu_ids=[
                0,
            ],
            batch_size=2,
            max_epochs=3,
            num_classes=80,
            load_pretrain=True,
            base_lr_per_img=0.001,
            train_image_dir='./data/test/images/image_detection/images',
            val_image_dir='./data/test/images/image_detection/images',
            train_ann=
            './data/test/images/image_detection/annotations/coco_sample.json',
            val_ann=
            './data/test/images/image_detection/annotations/coco_sample.json',
        )
        trainer = build_trainer(
            name=Trainers.tinynas_damoyolo, default_args=kwargs)
        trainer.train()
        trainer.evaluate(
            checkpoint_path=os.path.join(self.cache_path,
                                         'damoyolo_tinynasL25_S.pt'))

    @unittest.skip('multiGPU test is varified offline')
    def test_trainer_from_scratch_multiGPU(self):
        kwargs = dict(
            cfg_file=os.path.join(self.cache_path, 'configuration.json'),
            gpu_ids=[
                0,
                1,
            ],
            batch_size=32,
            max_epochs=3,
            num_classes=1,
            cache_path=self.cache_path,
            train_image_dir='./data/test/images/image_detection/images',
            val_image_dir='./data/test/images/image_detection/images',
            train_ann=
            './data/test/images/image_detection/annotations/coco_sample.json',
            val_ann=
            './data/test/images/image_detection/annotations/coco_sample.json')
        trainer = build_trainer(
            name=Trainers.tinynas_damoyolo, default_args=kwargs)
        trainer.train()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer_finetune_singleGPU(self):
        kwargs = dict(
            cfg_file=os.path.join(self.cache_path, 'configuration.json'),
            gpu_ids=[
                0,
            ],
            batch_size=16,
            max_epochs=3,
            num_classes=1,
            load_pretrain=True,
            pretrain_model=os.path.join(self.cache_path,
                                        'damoyolo_tinynasL25_S.pt'),
            cache_path=self.cache_path,
            train_image_dir='./data/test/images/image_detection/images',
            val_image_dir='./data/test/images/image_detection/images',
            train_ann=
            './data/test/images/image_detection/annotations/coco_sample.json',
            val_ann=
            './data/test/images/image_detection/annotations/coco_sample.json')
        trainer = build_trainer(
            name=Trainers.tinynas_damoyolo, default_args=kwargs)
        trainer.train()


if __name__ == '__main__':
    unittest.main()
